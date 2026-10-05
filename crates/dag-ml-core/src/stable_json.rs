//! JSON serialization with sorted map keys and unchanged struct field order.
//! Preserves existing digests while making embedded Values independent of
//! serde_json's optional `preserve_order` feature.
use serde::ser::{self, Serialize, SerializeMap, SerializeSeq, SerializeStruct};

pub(crate) fn to_vec<T: Serialize + ?Sized>(value: &T) -> serde_json::Result<Vec<u8>> {
    value.serialize(Serializer).map(String::into_bytes)
}

struct Serializer;
struct Array {
    values: Vec<String>,
    variant: Option<&'static str>,
}
struct Object {
    values: Vec<(String, String)>,
    key: Option<String>,
    sorted: bool,
    variant: Option<&'static str>,
}

fn tagged(name: &str, value: String) -> serde_json::Result<String> {
    Ok(format!("{{{}:{value}}}", serde_json::to_string(name)?))
}

macro_rules! scalar {
    ($($method:ident($ty:ty)),*) => {$(
        fn $method(self, value: $ty) -> serde_json::Result<String> {
            serde_json::to_string(&value)
        }
    )*};
}

impl ser::Serializer for Serializer {
    type Ok = String;
    type Error = serde_json::Error;
    type SerializeSeq = Array;
    type SerializeTuple = Array;
    type SerializeTupleStruct = Array;
    type SerializeTupleVariant = Array;
    type SerializeMap = Object;
    type SerializeStruct = Object;
    type SerializeStructVariant = Object;
    scalar!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_f32(f32),
        serialize_f64(f64),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8])
    );
    fn serialize_none(self) -> serde_json::Result<String> {
        self.serialize_unit()
    }
    fn serialize_unit(self) -> serde_json::Result<String> {
        Ok("null".into())
    }
    fn serialize_unit_struct(self, _: &'static str) -> serde_json::Result<String> {
        self.serialize_unit()
    }
    fn serialize_some<T: Serialize + ?Sized>(self, value: &T) -> serde_json::Result<String> {
        value.serialize(self)
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        value: &T,
    ) -> serde_json::Result<String> {
        value.serialize(self)
    }
    fn serialize_unit_variant(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
    ) -> serde_json::Result<String> {
        serde_json::to_string(name)
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
        value: &T,
    ) -> serde_json::Result<String> {
        tagged(name, value.serialize(self)?)
    }
    fn serialize_seq(self, _: Option<usize>) -> serde_json::Result<Array> {
        Ok(Array {
            values: Vec::new(),
            variant: None,
        })
    }
    fn serialize_tuple(self, len: usize) -> serde_json::Result<Array> {
        self.serialize_seq(Some(len))
    }
    fn serialize_tuple_struct(self, _: &'static str, len: usize) -> serde_json::Result<Array> {
        self.serialize_tuple(len)
    }
    fn serialize_tuple_variant(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
        _: usize,
    ) -> serde_json::Result<Array> {
        Ok(Array {
            values: Vec::new(),
            variant: Some(name),
        })
    }
    fn serialize_map(self, _: Option<usize>) -> serde_json::Result<Object> {
        Ok(Object {
            values: Vec::new(),
            key: None,
            sorted: true,
            variant: None,
        })
    }
    fn serialize_struct(self, _: &'static str, _: usize) -> serde_json::Result<Object> {
        Ok(Object {
            values: Vec::new(),
            key: None,
            sorted: false,
            variant: None,
        })
    }
    fn serialize_struct_variant(
        self,
        _: &'static str,
        _: u32,
        name: &'static str,
        len: usize,
    ) -> serde_json::Result<Object> {
        let mut object = self.serialize_struct(name, len)?;
        object.variant = Some(name);
        Ok(object)
    }
}

impl SerializeSeq for Array {
    type Ok = String;
    type Error = serde_json::Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, value: &T) -> serde_json::Result<()> {
        self.values.push(value.serialize(Serializer)?);
        Ok(())
    }
    fn end(self) -> serde_json::Result<String> {
        let value = format!("[{}]", self.values.join(","));
        self.variant
            .map_or(Ok(value.clone()), |name| tagged(name, value))
    }
}
macro_rules! array_impl {
    ($trait:ident, $method:ident) => {
        impl ser::$trait for Array {
            type Ok = String;
            type Error = serde_json::Error;
            fn $method<T: Serialize + ?Sized>(&mut self, value: &T) -> serde_json::Result<()> {
                SerializeSeq::serialize_element(self, value)
            }
            fn end(self) -> serde_json::Result<String> {
                SerializeSeq::end(self)
            }
        }
    };
}
array_impl!(SerializeTuple, serialize_element);
array_impl!(SerializeTupleStruct, serialize_field);
array_impl!(SerializeTupleVariant, serialize_field);

impl Object {
    fn finish(mut self) -> serde_json::Result<String> {
        if self.sorted {
            self.values.sort_by(|a, b| a.0.cmp(&b.0));
        }
        let fields = self
            .values
            .into_iter()
            .map(|(key, value)| Ok(format!("{}:{value}", serde_json::to_string(&key)?)))
            .collect::<serde_json::Result<Vec<_>>>()?;
        let value = format!("{{{}}}", fields.join(","));
        self.variant
            .map_or(Ok(value.clone()), |name| tagged(name, value))
    }
}
impl SerializeMap for Object {
    type Ok = String;
    type Error = serde_json::Error;
    fn serialize_key<T: Serialize + ?Sized>(&mut self, key: &T) -> serde_json::Result<()> {
        let raw = key.serialize(Serializer)?;
        self.key = Some(if raw.starts_with('"') {
            serde_json::from_str(&raw)?
        } else {
            raw
        });
        Ok(())
    }
    fn serialize_value<T: Serialize + ?Sized>(&mut self, value: &T) -> serde_json::Result<()> {
        let key = self
            .key
            .take()
            .ok_or_else(|| <serde_json::Error as ser::Error>::custom("map value without key"))?;
        self.values.push((key, value.serialize(Serializer)?));
        Ok(())
    }
    fn end(self) -> serde_json::Result<String> {
        self.finish()
    }
}
impl SerializeStruct for Object {
    type Ok = String;
    type Error = serde_json::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> serde_json::Result<()> {
        self.values.push((key.into(), value.serialize(Serializer)?));
        Ok(())
    }
    fn end(self) -> serde_json::Result<String> {
        self.finish()
    }
}
impl ser::SerializeStructVariant for Object {
    type Ok = String;
    type Error = serde_json::Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> serde_json::Result<()> {
        SerializeStruct::serialize_field(self, key, value)
    }
    fn end(self) -> serde_json::Result<String> {
        self.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sorts_values_without_changing_struct_order() {
        #[derive(serde::Serialize)]
        struct Example {
            z: u64,
            a: serde_json::Value,
        }
        let value = Example {
            z: 1,
            a: serde_json::from_str(r#"{"z":2,"a":{"z":3,"a":4}}"#).unwrap(),
        };
        assert_eq!(
            String::from_utf8(to_vec(&value).unwrap()).unwrap(),
            r#"{"z":1,"a":{"a":{"a":4,"z":3},"z":2}}"#
        );
    }
}
