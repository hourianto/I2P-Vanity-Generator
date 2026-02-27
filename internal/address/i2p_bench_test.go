package address

import "testing"

var sinkI2PMatch bool

func BenchmarkI2PMutateAndCheck(b *testing.B) {
	candAny, err := I2PScheme{}.NewCandidate()
	if err != nil {
		b.Fatal(err)
	}
	cand := candAny.(*I2PCandidate)
	prefix := "abcde"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkI2PMatch = cand.MutateAndCheck(uint64(i), prefix)
	}
}
