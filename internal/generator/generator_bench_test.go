package generator

import (
	"context"
	"testing"
	"time"

	"github.com/go-i2p/i2p-vanitygen/internal/address"
)

func runI2PThroughputSample(cores int, d time.Duration) float64 {
	g := New(address.I2PScheme{}, "zzzzzzzz", cores, false, 0)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	resultCh, statsCh := g.Start(ctx)

	go func() {
		for range resultCh {
		}
	}()

	deadline := time.NewTimer(d)
	defer deadline.Stop()

	var latest Stats
	for {
		select {
		case st, ok := <-statsCh:
			if !ok {
				if latest.Elapsed > 0 {
					return float64(latest.Checked) / latest.Elapsed.Seconds()
				}
				return 0
			}
			latest = st
		case <-deadline.C:
			cancel()
			for st := range statsCh {
				latest = st
			}
			if latest.Elapsed > 0 {
				return float64(latest.Checked) / latest.Elapsed.Seconds()
			}
			return 0
		}
	}
}

func BenchmarkI2PGeneratorThroughput1Core(b *testing.B) {
	for i := 0; i < b.N; i++ {
		kps := runI2PThroughputSample(1, 600*time.Millisecond)
		b.ReportMetric(kps, "keys/sec")
	}
}

func BenchmarkI2PGeneratorThroughput8Cores(b *testing.B) {
	for i := 0; i < b.N; i++ {
		kps := runI2PThroughputSample(8, 600*time.Millisecond)
		b.ReportMetric(kps, "keys/sec")
	}
}
