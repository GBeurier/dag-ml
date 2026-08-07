//! Independent Rust oracle for DAG-ML canonical fingerprint profiles.
//!
//! This crate is test-only and intentionally does not import `dag-ml-core` or
//! the Python oracle.  It parses JSON itself so integer and binary64 tokens stay
//! distinguishable, then implements DAG-ML TCV1 and the restricted RFC 8785/JCS
//! domain used by the ordered-search-space contract.

use sha2::{Digest, Sha256};
use std::collections::HashSet;
use unicode_normalization::UnicodeNormalization;

const TCV1_PREFIX: &[u8] = b"DAGML-TCV1\0";
const JCS_SAFE_INTEGER_MAX: u64 = (1_u64 << 53) - 1;

/// A strict JSON value retaining the integer-versus-binary64 token kind.
#[derive(Clone, Debug, PartialEq)]
pub enum CanonicalValue {
    Null,
    Bool(bool),
    Signed(i64),
    Unsigned(u64),
    Binary64(f64),
    String(String),
    Array(Vec<CanonicalValue>),
    Object(Vec<(String, CanonicalValue)>),
}

/// Parse one strict UTF-8 JSON document without losing numeric token kinds.
pub fn parse_strict_json(input: &str) -> Result<CanonicalValue, String> {
    let mut parser = Parser { input, offset: 0 };
    let value = parser.parse_value()?;
    parser.skip_whitespace();
    if parser.offset != input.len() {
        return Err(parser.error("trailing data after the JSON value"));
    }
    Ok(value)
}

/// Return the domain-separated DAG-ML TCV1 preimage.
pub fn tcv1_preimage(value: &CanonicalValue) -> Result<Vec<u8>, String> {
    let mut output = Vec::new();
    output.extend_from_slice(TCV1_PREFIX);
    encode_tcv1(value, &mut output)?;
    Ok(output)
}

/// Return the lowercase SHA-256 digest of the DAG-ML TCV1 preimage.
pub fn tcv1_sha256(value: &CanonicalValue) -> Result<String, String> {
    Ok(sha256_hex(&tcv1_preimage(value)?))
}

/// Canonicalize the restricted JCS domain used by OrderedSearchSpaceSpec V1.
///
/// Binary64 values are wire strings in that contract.  Raw JSON floating-point
/// tokens are rejected so this oracle cannot silently use Rust formatting where
/// RFC 8785 requires ECMAScript number serialization.
pub fn restricted_jcs_bytes(value: &CanonicalValue) -> Result<Vec<u8>, String> {
    let mut output = String::new();
    encode_restricted_jcs(value, &mut output)?;
    Ok(output.into_bytes())
}

/// Return the `sha256:`-prefixed restricted-JCS fingerprint.
pub fn restricted_jcs_fingerprint(value: &CanonicalValue) -> Result<String, String> {
    Ok(format!(
        "sha256:{}",
        sha256_hex(&restricted_jcs_bytes(value)?)
    ))
}

/// Lowercase hexadecimal bytes, exposed for the tiny command-line adapter.
pub fn lowercase_hex(bytes: &[u8]) -> String {
    let mut output = String::with_capacity(bytes.len() * 2);
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn sha256_hex(bytes: &[u8]) -> String {
    lowercase_hex(&Sha256::digest(bytes))
}

fn encode_tcv1(value: &CanonicalValue, output: &mut Vec<u8>) -> Result<(), String> {
    match value {
        CanonicalValue::Null => output.push(b'N'),
        CanonicalValue::Bool(false) => output.push(b'F'),
        CanonicalValue::Bool(true) => output.push(b'T'),
        CanonicalValue::Signed(number) => encode_tcv1_integer(number.to_string(), output)?,
        CanonicalValue::Unsigned(number) => encode_tcv1_integer(number.to_string(), output)?,
        CanonicalValue::Binary64(number) => {
            if !number.is_finite() {
                return Err("TCV1 binary64 value must be finite".to_string());
            }
            output.push(b'D');
            let canonical = if *number == 0.0 { 0.0 } else { *number };
            output.extend_from_slice(&canonical.to_be_bytes());
        }
        CanonicalValue::String(text) => encode_tcv1_string(text, output)?,
        CanonicalValue::Array(members) => {
            output.push(b'A');
            encode_length(members.len(), output)?;
            for member in members {
                encode_tcv1(member, output)?;
            }
        }
        CanonicalValue::Object(members) => {
            let mut normalized = Vec::with_capacity(members.len());
            let mut keys = HashSet::with_capacity(members.len());
            for (key, member) in members {
                let key = key.nfc().collect::<String>();
                if !keys.insert(key.as_bytes().to_vec()) {
                    return Err("TCV1 object has NFC-colliding keys".to_string());
                }
                normalized.push((key, member));
            }
            normalized.sort_by(|left, right| left.0.as_bytes().cmp(right.0.as_bytes()));

            output.push(b'O');
            encode_length(normalized.len(), output)?;
            for (key, member) in normalized {
                encode_tcv1_string(&key, output)?;
                encode_tcv1(member, output)?;
            }
        }
    }
    Ok(())
}

fn encode_tcv1_integer(payload: String, output: &mut Vec<u8>) -> Result<(), String> {
    output.push(b'I');
    encode_length(payload.len(), output)?;
    output.extend_from_slice(payload.as_bytes());
    Ok(())
}

fn encode_tcv1_string(text: &str, output: &mut Vec<u8>) -> Result<(), String> {
    let normalized = text.nfc().collect::<String>();
    output.push(b'S');
    encode_length(normalized.len(), output)?;
    output.extend_from_slice(normalized.as_bytes());
    Ok(())
}

fn encode_length(length: usize, output: &mut Vec<u8>) -> Result<(), String> {
    let length = u64::try_from(length).map_err(|_| "TCV1 length exceeds u64".to_string())?;
    output.extend_from_slice(&length.to_be_bytes());
    Ok(())
}

fn encode_restricted_jcs(value: &CanonicalValue, output: &mut String) -> Result<(), String> {
    match value {
        CanonicalValue::Null => output.push_str("null"),
        CanonicalValue::Bool(false) => output.push_str("false"),
        CanonicalValue::Bool(true) => output.push_str("true"),
        CanonicalValue::Signed(number) => {
            if *number < 0 || (*number as u64) > JCS_SAFE_INTEGER_MAX {
                return Err("restricted JCS structural integers must be in 0..2^53-1".to_string());
            }
            output.push_str(&number.to_string());
        }
        CanonicalValue::Unsigned(number) => {
            if *number > JCS_SAFE_INTEGER_MAX {
                return Err("restricted JCS structural integers must be in 0..2^53-1".to_string());
            }
            output.push_str(&number.to_string());
        }
        CanonicalValue::Binary64(_) => {
            return Err(
                "restricted JCS requires binary64-derived values to be tagged strings".to_string(),
            );
        }
        CanonicalValue::String(text) => encode_jcs_string(text, output),
        CanonicalValue::Array(members) => {
            output.push('[');
            for (index, member) in members.iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                encode_restricted_jcs(member, output)?;
            }
            output.push(']');
        }
        CanonicalValue::Object(members) => {
            let mut sorted = members.iter().collect::<Vec<_>>();
            sorted.sort_by_key(|member| utf16_units(&member.0));
            output.push('{');
            for (index, (key, member)) in sorted.into_iter().enumerate() {
                if index != 0 {
                    output.push(',');
                }
                encode_jcs_string(key, output);
                output.push(':');
                encode_restricted_jcs(member, output)?;
            }
            output.push('}');
        }
    }
    Ok(())
}

fn utf16_units(value: &str) -> Vec<u16> {
    value.encode_utf16().collect()
}

fn encode_jcs_string(value: &str, output: &mut String) {
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{0008}' => output.push_str("\\b"),
            '\u{0009}' => output.push_str("\\t"),
            '\u{000a}' => output.push_str("\\n"),
            '\u{000c}' => output.push_str("\\f"),
            '\u{000d}' => output.push_str("\\r"),
            '\u{0000}'..='\u{001f}' => {
                output.push_str(&format!("\\u{:04x}", character as u32));
            }
            _ => output.push(character),
        }
    }
    output.push('"');
}

struct Parser<'a> {
    input: &'a str,
    offset: usize,
}

impl Parser<'_> {
    fn parse_value(&mut self) -> Result<CanonicalValue, String> {
        self.skip_whitespace();
        match self.peek_byte() {
            Some(b'n') => {
                self.consume_literal("null")?;
                Ok(CanonicalValue::Null)
            }
            Some(b'f') => {
                self.consume_literal("false")?;
                Ok(CanonicalValue::Bool(false))
            }
            Some(b't') => {
                self.consume_literal("true")?;
                Ok(CanonicalValue::Bool(true))
            }
            Some(b'"') => self.parse_string().map(CanonicalValue::String),
            Some(b'[') => self.parse_array(),
            Some(b'{') => self.parse_object(),
            Some(b'-' | b'0'..=b'9') => self.parse_number(),
            Some(_) => Err(self.error("unexpected token")),
            None => Err(self.error("unexpected end of input")),
        }
    }

    fn parse_array(&mut self) -> Result<CanonicalValue, String> {
        self.expect_byte(b'[')?;
        self.skip_whitespace();
        let mut members = Vec::new();
        if self.consume_byte(b']') {
            return Ok(CanonicalValue::Array(members));
        }
        loop {
            members.push(self.parse_value()?);
            self.skip_whitespace();
            if self.consume_byte(b']') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(CanonicalValue::Array(members))
    }

    fn parse_object(&mut self) -> Result<CanonicalValue, String> {
        self.expect_byte(b'{')?;
        self.skip_whitespace();
        let mut members = Vec::new();
        let mut keys = HashSet::new();
        if self.consume_byte(b'}') {
            return Ok(CanonicalValue::Object(members));
        }
        loop {
            self.skip_whitespace();
            if self.peek_byte() != Some(b'"') {
                return Err(self.error("JSON object key must be a string"));
            }
            let key = self.parse_string()?;
            if !keys.insert(key.clone()) {
                return Err(self.error("duplicate JSON object key"));
            }
            self.skip_whitespace();
            self.expect_byte(b':')?;
            let member = self.parse_value()?;
            members.push((key, member));
            self.skip_whitespace();
            if self.consume_byte(b'}') {
                break;
            }
            self.expect_byte(b',')?;
        }
        Ok(CanonicalValue::Object(members))
    }

    fn parse_number(&mut self) -> Result<CanonicalValue, String> {
        let start = self.offset;
        let negative = self.consume_byte(b'-');
        match self.peek_byte() {
            Some(b'0') => {
                self.offset += 1;
                if matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                    return Err(self.error("JSON number has a leading zero"));
                }
            }
            Some(b'1'..=b'9') => {
                self.offset += 1;
                while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                    self.offset += 1;
                }
            }
            _ => return Err(self.error("invalid JSON number integer part")),
        }

        let mut binary64 = false;
        if self.consume_byte(b'.') {
            binary64 = true;
            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("JSON fraction must contain a digit"));
            }
            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }
        if matches!(self.peek_byte(), Some(b'e' | b'E')) {
            binary64 = true;
            self.offset += 1;
            if matches!(self.peek_byte(), Some(b'+' | b'-')) {
                self.offset += 1;
            }
            if !matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                return Err(self.error("JSON exponent must contain a digit"));
            }
            while matches!(self.peek_byte(), Some(b'0'..=b'9')) {
                self.offset += 1;
            }
        }

        let token = &self.input[start..self.offset];
        if binary64 {
            let value = token
                .parse::<f64>()
                .map_err(|_| self.error("invalid binary64 JSON number"))?;
            if !value.is_finite() {
                return Err(self.error("JSON number is outside finite binary64 range"));
            }
            return Ok(CanonicalValue::Binary64(value));
        }
        if negative {
            let value = token
                .parse::<i64>()
                .map_err(|_| self.error("integer is below the TCV1 i64 minimum"))?;
            Ok(CanonicalValue::Signed(value))
        } else {
            let value = token
                .parse::<u64>()
                .map_err(|_| self.error("integer is above the TCV1 u64 maximum"))?;
            Ok(CanonicalValue::Unsigned(value))
        }
    }

    fn parse_string(&mut self) -> Result<String, String> {
        self.expect_byte(b'"')?;
        let mut output = String::new();
        loop {
            let byte = self
                .peek_byte()
                .ok_or_else(|| self.error("unterminated JSON string"))?;
            match byte {
                b'"' => {
                    self.offset += 1;
                    return Ok(output);
                }
                b'\\' => {
                    self.offset += 1;
                    let escape = self
                        .peek_byte()
                        .ok_or_else(|| self.error("unterminated JSON escape"))?;
                    self.offset += 1;
                    match escape {
                        b'"' => output.push('"'),
                        b'\\' => output.push('\\'),
                        b'/' => output.push('/'),
                        b'b' => output.push('\u{0008}'),
                        b'f' => output.push('\u{000c}'),
                        b'n' => output.push('\n'),
                        b'r' => output.push('\r'),
                        b't' => output.push('\t'),
                        b'u' => self.parse_unicode_escape(&mut output)?,
                        _ => return Err(self.error("invalid JSON string escape")),
                    }
                }
                0x00..=0x1f => {
                    return Err(self.error("unescaped control character in JSON string"));
                }
                _ => {
                    let character = self.input[self.offset..]
                        .chars()
                        .next()
                        .ok_or_else(|| self.error("invalid UTF-8 in JSON string"))?;
                    output.push(character);
                    self.offset += character.len_utf8();
                }
            }
        }
    }

    fn parse_unicode_escape(&mut self, output: &mut String) -> Result<(), String> {
        let first = self.parse_hex_quad()?;
        let scalar = if (0xd800..=0xdbff).contains(&first) {
            if !self.consume_bytes(b"\\u") {
                return Err(self.error("high surrogate is not followed by a low surrogate"));
            }
            let second = self.parse_hex_quad()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(self.error("high surrogate is not followed by a low surrogate"));
            }
            0x10000 + (((first as u32 - 0xd800) << 10) | (second as u32 - 0xdc00))
        } else if (0xdc00..=0xdfff).contains(&first) {
            return Err(self.error("unpaired low surrogate in JSON string"));
        } else {
            first as u32
        };
        let character = char::from_u32(scalar)
            .ok_or_else(|| self.error("invalid Unicode scalar in JSON string"))?;
        output.push(character);
        Ok(())
    }

    fn parse_hex_quad(&mut self) -> Result<u16, String> {
        let mut value = 0_u16;
        for _ in 0..4 {
            let digit = self
                .peek_byte()
                .and_then(hex_digit)
                .ok_or_else(|| self.error("invalid four-digit Unicode escape"))?;
            self.offset += 1;
            value = (value << 4) | digit as u16;
        }
        Ok(value)
    }

    fn consume_literal(&mut self, literal: &str) -> Result<(), String> {
        if self.input[self.offset..].starts_with(literal) {
            self.offset += literal.len();
            Ok(())
        } else {
            Err(self.error("invalid JSON literal"))
        }
    }

    fn skip_whitespace(&mut self) {
        while matches!(self.peek_byte(), Some(b' ' | b'\n' | b'\r' | b'\t')) {
            self.offset += 1;
        }
    }

    fn expect_byte(&mut self, expected: u8) -> Result<(), String> {
        self.skip_whitespace();
        if self.consume_byte(expected) {
            Ok(())
        } else {
            Err(self.error(&format!("expected `{}`", expected as char)))
        }
    }

    fn consume_byte(&mut self, expected: u8) -> bool {
        if self.peek_byte() == Some(expected) {
            self.offset += 1;
            true
        } else {
            false
        }
    }

    fn consume_bytes(&mut self, expected: &[u8]) -> bool {
        if self.input.as_bytes()[self.offset..].starts_with(expected) {
            self.offset += expected.len();
            true
        } else {
            false
        }
    }

    fn peek_byte(&self) -> Option<u8> {
        self.input.as_bytes().get(self.offset).copied()
    }

    fn error(&self, message: &str) -> String {
        format!("{message} at UTF-8 byte {}", self.offset)
    }
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_preserves_integer_and_binary64_token_kinds() {
        assert_eq!(parse_strict_json("2").unwrap(), CanonicalValue::Unsigned(2));
        assert_eq!(
            parse_strict_json("2.0").unwrap(),
            CanonicalValue::Binary64(2.0)
        );
        assert_eq!(parse_strict_json("-0").unwrap(), CanonicalValue::Signed(0));
        assert!(matches!(
            parse_strict_json("-0.0").unwrap(),
            CanonicalValue::Binary64(value) if value.is_sign_negative()
        ));
    }

    #[test]
    fn parser_rejects_duplicate_members_surrogates_and_out_of_range_numbers() {
        assert!(parse_strict_json(r#"{"a":1,"a":2}"#).is_err());
        assert!(parse_strict_json(r#""\ud800""#).is_err());
        assert!(parse_strict_json("18446744073709551616").is_err());
        assert!(parse_strict_json("-9223372036854775809").is_err());
        assert!(parse_strict_json("1e400").is_err());
    }

    #[test]
    fn profiles_keep_their_key_orders_disjoint() {
        let value = parse_strict_json(r#"{"\ue000":1,"\ud800\udc00":2}"#).unwrap();
        let tcv1 = tcv1_preimage(&value).unwrap();
        let jcs = String::from_utf8(restricted_jcs_bytes(&value).unwrap()).unwrap();
        let private_use = tcv1
            .windows(3)
            .position(|window| window == [0xee, 0x80, 0x80]);
        let supplementary = tcv1
            .windows(4)
            .position(|window| window == [0xf0, 0x90, 0x80, 0x80]);
        assert!(private_use < supplementary);
        assert_eq!(jcs, "{\"𐀀\":2,\"\":1}");
    }

    #[test]
    fn tcv1_rejects_nfc_colliding_keys_but_jcs_keeps_them_distinct() {
        let value = parse_strict_json(r#"{"é":1,"e\u0301":2}"#).unwrap();
        assert!(tcv1_preimage(&value).is_err());
        assert_eq!(
            String::from_utf8(restricted_jcs_bytes(&value).unwrap()).unwrap(),
            "{\"é\":2,\"é\":1}"
        );
    }
}
