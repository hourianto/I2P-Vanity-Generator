//go:build darwin && cgo

package gpu

import (
	"testing"

	"github.com/go-i2p/i2p-vanitygen/internal/address"
)

const (
	benchI2PBatchSize   = uint64(1 << 20) // 1,048,576 hashes
	benchTorV3BatchSize = uint64(1 << 16) // 65,536 pubkeys
)

var (
	sinkGPUBatchResult BatchResult
	sinkGPUChecked     uint64
)

func BenchmarkMetalI2PRunBatch(b *testing.B) {
	if !Available() {
		b.Skip("Metal GPU not available")
	}

	candAny, err := address.I2PScheme{}.NewCandidate()
	if err != nil {
		b.Fatal(err)
	}
	i2pCand := candAny.(*address.I2PCandidate)

	worker, err := NewWorker(WorkerConfig{
		DeviceIndex:  0,
		DestTemplate: i2pCand.Raw(),
		Prefix:       "zzzzzzzzzzzz",
		BatchSize:    benchI2PBatchSize,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer worker.Close()

	var totalChecked uint64
	counter := uint64(0)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := worker.RunBatch(counter)
		if err != nil {
			b.Fatalf("run batch: %v", err)
		}
		totalChecked += result.Checked
		counter += result.Checked
		sinkGPUBatchResult = result
	}
	b.StopTimer()

	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(float64(totalChecked)/elapsed, "keys/sec")
	}
	sinkGPUChecked = totalChecked
}

func BenchmarkMetalTorV3RunBatch(b *testing.B) {
	if !Available() {
		b.Skip("Metal GPU not available")
	}

	pubkeys := precomputeTorV3Pubkeys(b, benchTorV3BatchSize)

	worker, err := NewTorV3Worker(TorV3WorkerConfig{
		DeviceIndex: 0,
		Prefix:      "zzzzzzzzzz",
		BatchSize:   benchTorV3BatchSize,
	})
	if err != nil {
		b.Fatal(err)
	}
	defer worker.Close()

	var totalChecked uint64

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		result, err := worker.RunBatch(pubkeys, benchTorV3BatchSize)
		if err != nil {
			b.Fatalf("run batch: %v", err)
		}
		totalChecked += result.Checked
		sinkGPUBatchResult = result
	}
	b.StopTimer()

	elapsed := b.Elapsed().Seconds()
	if elapsed > 0 {
		b.ReportMetric(float64(totalChecked)/elapsed, "keys/sec")
	}
	sinkGPUChecked = totalChecked
}

func precomputeTorV3Pubkeys(b *testing.B, keyCount uint64) []byte {
	b.Helper()

	cand, err := address.NewTorV3Candidate()
	if err != nil {
		b.Fatal(err)
	}

	buf := make([]byte, keyCount*32)
	for i := uint64(0); i < keyCount; i++ {
		copy(buf[i*32:(i+1)*32], cand.PublicKeyBytes())
		cand.Advance()
	}
	return buf
}
