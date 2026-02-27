package base32check

// HasPrefixLowerNoPad reports whether the lowercase base32 (RFC4648, no padding)
// encoding of data starts with prefix. Prefix must be ASCII base32 chars.
func HasPrefixLowerNoPad(data []byte, prefix string) bool {
	if len(prefix) == 0 {
		return true
	}

	maxChars := (len(data)*8 + 4) / 5
	if len(prefix) > maxChars {
		return false
	}

	const alphabet = "abcdefghijklmnopqrstuvwxyz234567"
	for i := 0; i < len(prefix); i++ {
		bitOffset := i * 5
		byteIdx := bitOffset / 8
		bitIdx := bitOffset % 8

		var val byte
		if bitIdx <= 3 {
			val = (data[byteIdx] >> (3 - bitIdx)) & 0x1f
		} else {
			val = (data[byteIdx] << (bitIdx - 3)) & 0x1f
			if byteIdx+1 < len(data) {
				val |= data[byteIdx+1] >> (11 - bitIdx)
			}
		}

		want := prefix[i]
		if want >= 'A' && want <= 'Z' {
			want = want + ('a' - 'A')
		}
		if alphabet[val] != want {
			return false
		}
	}

	return true
}
