package address

import "testing"

var sinkTorMatch bool

func BenchmarkTorV3CheckPrefix(b *testing.B) {
	cand, err := NewTorV3Candidate()
	if err != nil {
		b.Fatal(err)
	}
	prefix := "abcde"

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		sinkTorMatch = cand.CheckPrefix(prefix)
		cand.Advance()
	}
}
