//go:build darwin && cgo

package gpu

import (
	"testing"

	"github.com/go-i2p/i2p-vanitygen/internal/address"
)

func TestMetalI2PBatchExactPrefixMatch(t *testing.T) {
	if !Available() {
		t.Skip("Metal GPU not available")
	}

	candAny, err := address.I2PScheme{}.NewCandidate()
	if err != nil {
		t.Fatal(err)
	}
	cand := candAny.(*address.I2PCandidate)
	cand.Dest.MutateEncryptionKey(0)
	fullPrefix := cand.Address()

	worker, err := NewWorker(WorkerConfig{
		DeviceIndex:  0,
		DestTemplate: cand.Raw(),
		Prefix:       fullPrefix,
		BatchSize:    1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	result, err := worker.RunBatch(0)
	if err != nil {
		t.Fatal(err)
	}
	if !result.Found {
		t.Fatalf("expected match for exact full prefix, got %+v", result)
	}
	if result.MatchCounter != 0 {
		t.Fatalf("expected counter 0, got %d", result.MatchCounter)
	}
	if result.Checked != 1 {
		t.Fatalf("expected checked=1, got %d", result.Checked)
	}
}

func TestMetalI2PBatchExactPrefixMismatch(t *testing.T) {
	if !Available() {
		t.Skip("Metal GPU not available")
	}

	candAny, err := address.I2PScheme{}.NewCandidate()
	if err != nil {
		t.Fatal(err)
	}
	cand := candAny.(*address.I2PCandidate)
	cand.Dest.MutateEncryptionKey(0)
	prefix := cand.Address()

	last := prefix[len(prefix)-1]
	if last == 'a' {
		last = 'b'
	} else {
		last = 'a'
	}
	prefix = prefix[:len(prefix)-1] + string(last)

	worker, err := NewWorker(WorkerConfig{
		DeviceIndex:  0,
		DestTemplate: cand.Raw(),
		Prefix:       prefix,
		BatchSize:    1,
	})
	if err != nil {
		t.Fatal(err)
	}
	defer worker.Close()

	result, err := worker.RunBatch(0)
	if err != nil {
		t.Fatal(err)
	}
	if result.Found {
		t.Fatalf("expected no match for mismatch prefix, got %+v", result)
	}
	if result.Checked != 1 {
		t.Fatalf("expected checked=1, got %d", result.Checked)
	}
}
