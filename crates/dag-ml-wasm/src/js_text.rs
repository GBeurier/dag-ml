//! JavaScript text ingress and failure rendering at the WASM boundary.
use crate::{callback_utf16_string, js_core_error, CoreDagMlError};
use js_sys::JsString;
use wasm_bindgen::prelude::*;

pub(crate) fn text_argument(value: &JsString, name: &str) -> Result<String, JsValue> {
    match callback_utf16_string(value.as_ref()) {
        Ok(Some(text)) => Ok(text),
        Ok(None) => Err(js_core_error(CoreDagMlError::RuntimeValidation(format!(
            "argument `{name}` must be a string"
        )))),
        Err(()) => Err(js_core_error(CoreDagMlError::RuntimeValidation(format!(
            "argument `{name}` is not valid UTF-16"
        )))),
    }
}

pub(crate) fn optional_text_argument(
    value: Option<JsString>,
    name: &str,
) -> Result<Option<String>, JsValue> {
    value
        .as_ref()
        .map(|value| text_argument(value, name))
        .transpose()
}

use js_sys::{Array, Function, Reflect, Symbol};
use wasm_bindgen::JsCast;

const UNRENDERABLE: &str = "<unrenderable JS value>";
const UNREADABLE: &str = "<unreadable>";
const NODE_LIMIT: usize = 1024;

fn rendered_string(value: &JsValue) -> Option<String> {
    let text = value.as_string()?;
    let js: &JsString = value.unchecked_ref();
    if !text.contains('\u{fffd}') || js.is_valid_utf16() {
        return Some(text);
    }
    let units: Vec<u16> = js.iter().collect();
    Some(render_malformed_units(&units))
}

fn render_malformed_units(units: &[u16]) -> String {
    let mut result = String::from("ill_formed_utf16(\"");
    let mut index = 0;
    while index < units.len() {
        let unit = units[index];
        if (0xd800..=0xdbff).contains(&unit)
            && index + 1 < units.len()
            && (0xdc00..=0xdfff).contains(&units[index + 1])
        {
            let code =
                0x10000 + (((unit as u32) - 0xd800) << 10) + ((units[index + 1] as u32) - 0xdc00);
            if let Some(ch) = char::from_u32(code) {
                result.push(ch);
            }
            index += 2;
            continue;
        }
        match unit {
            0xd800..=0xdfff => result.push_str(&format!("\\u{unit:04x}")),
            0x22 => result.push_str("\\\""),
            0x5c => result.push_str("\\\\"),
            _ => {
                if let Some(ch) = char::from_u32(unit as u32) {
                    result.push(ch);
                }
            }
        }
        index += 1;
    }
    result.push_str("\")");
    result
}

fn property(value: &JsValue, key: &str) -> Result<JsValue, JsValue> {
    Reflect::get(value, &JsValue::from_str(key))
}

fn builtin_prototype_method(constructor: &str, method: &str) -> Option<Function> {
    let global: JsValue = js_sys::global().into();
    let ctor = property(&global, constructor).ok()?;
    let prototype = property(&ctor, "prototype").ok()?;
    let f = property(&prototype, method).ok()?;
    if !f.is_function() {
        return None;
    }
    Some(f.unchecked_into())
}

fn call_method(value: &JsValue, constructor: &str, method: &str) -> Option<JsValue> {
    Reflect::apply(
        &builtin_prototype_method(constructor, method)?,
        value,
        &Array::new(),
    )
    .ok()
}

fn symbol_inner(value: &JsValue) -> Option<String> {
    let rendered = call_method(value, "Symbol", "toString")?;
    let js: &JsString = rendered.unchecked_ref();
    if rendered.as_string()?.contains('\u{fffd}') && !js.is_valid_utf16() {
        return rendered_string(&rendered);
    }
    let global: JsValue = js_sys::global().into();
    let constructor = property(&global, "Object").ok()?;
    if !constructor.is_function() {
        return None;
    }
    let constructor: Function = constructor.unchecked_into();
    let boxed = Reflect::apply(&constructor, &JsValue::UNDEFINED, &Array::of1(value)).ok()?;
    let description = property(&boxed, "description").ok()?;
    if description.is_null_or_undefined() {
        Some("Symbol".into())
    } else {
        rendered_string(&rendered)
    }
}

fn is_error(value: &JsValue) -> bool {
    let global: JsValue = js_sys::global().into();
    let Some(ctor) = property(&global, "Error").ok() else {
        return false;
    };
    let Ok(has_instance) = Reflect::get(&ctor, Symbol::has_instance().as_ref()) else {
        return false;
    };
    if !has_instance.is_function() {
        return false;
    }
    let method: Function = has_instance.unchecked_into();
    Reflect::apply(&method, &ctor, &Array::of1(value))
        .ok()
        .and_then(|answer| answer.as_bool())
        .unwrap_or(false)
}

fn error_field(value: &JsValue, name: &str) -> String {
    match property(value, name) {
        Err(_) => UNREADABLE.to_string(),
        Ok(field) if field.is_string() => {
            rendered_string(&field).unwrap_or_else(|| UNREADABLE.into())
        }
        Ok(field) if field.is_object() || field.is_function() => UNRENDERABLE.into(),
        Ok(field) if field.is_symbol() => {
            symbol_inner(&field).unwrap_or_else(|| UNRENDERABLE.into())
        }
        Ok(field) => primitive_inner(&field),
    }
}

fn primitive_inner(value: &JsValue) -> String {
    let text = format!("{value:?}");
    text.strip_prefix("JsValue(")
        .and_then(|s| s.strip_suffix(')'))
        .unwrap_or(&text)
        .to_string()
}

fn render_inner(value: &JsValue, ancestors: &mut Vec<JsValue>, budget: &mut usize) -> String {
    if *budget == 0 {
        return "...".into();
    }
    *budget -= 1;
    if value.is_string() {
        let s = rendered_string(value).unwrap_or_else(|| UNRENDERABLE.into());
        let malformed = value.as_string().is_some_and(|text| {
            text.contains('\u{fffd}') && !value.unchecked_ref::<JsString>().is_valid_utf16()
        });
        return if malformed { s } else { format!("\"{s}\"") };
    }
    if value.is_symbol() {
        return symbol_inner(value).unwrap_or_else(|| UNRENDERABLE.into());
    }
    if value.is_function() {
        let name = property(value, "name")
            .ok()
            .and_then(|x| rendered_string(&x));
        return match name {
            Some(name) if !name.is_empty() => format!("Function({name})"),
            _ => "Function".into(),
        };
    }
    if !value.is_object() {
        return primitive_inner(value);
    }
    let Some(class_value) = call_method(value, "Object", "toString") else {
        return UNRENDERABLE.into();
    };
    let Some(class_text) = class_value.as_string() else {
        return UNRENDERABLE.into();
    };
    let Some(class) = class_text
        .strip_prefix("[object ")
        .and_then(|x| x.strip_suffix(']'))
    else {
        return UNRENDERABLE.into();
    };
    if class_text.contains('\u{fffd}') && !class_value.unchecked_ref::<JsString>().is_valid_utf16()
    {
        let units: Vec<u16> = class_value.unchecked_ref::<JsString>().iter().collect();
        if units.len() < 9 {
            return UNRENDERABLE.into();
        }
        return render_malformed_units(&units[8..units.len() - 1]);
    }
    if class == "Array" {
        if ancestors.iter().any(|ancestor| ancestor == value) {
            return "...".into();
        }
        let Some(length) = property(value, "length").ok().and_then(|x| x.as_f64()) else {
            return UNRENDERABLE.into();
        };
        ancestors.push(value.clone());
        let mut values = Vec::new();
        for index in 0..(length as usize) {
            if *budget == 0 {
                values.push("...".into());
                break;
            }
            let item = Reflect::get(value, &JsValue::from_f64(index as f64));
            values.push(match item {
                Ok(item) => render_inner(&item, ancestors, budget),
                Err(_) => {
                    *budget -= 1;
                    UNREADABLE.into()
                }
            });
        }
        ancestors.pop();
        return format!("[{}]", values.join(", "));
    }
    if class == "Object" {
        return match js_sys::JSON::stringify(value) {
            Ok(json) => format!(
                "Object({})",
                rendered_string(json.as_ref()).unwrap_or_else(|| UNRENDERABLE.into())
            ),
            Err(_) => "Object".into(),
        };
    }
    // An Error renders as `name: message` only. Its `stack` is never read: host
    // frames can embed inlined module URLs (a single-file WASM `data:` URL is
    // megabytes per frame) and are not part of the owner failure record.
    if is_error(value) {
        return format!(
            "{}: {}",
            error_field(value, "name"),
            error_field(value, "message")
        );
    }
    class.into()
}

pub(crate) fn js_failure_text(value: &JsValue) -> String {
    let mut budget = NODE_LIMIT;
    format!(
        "JsValue({})",
        render_inner(value, &mut Vec::new(), &mut budget)
    )
}
