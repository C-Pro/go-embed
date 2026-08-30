package engine_test

import (
	"testing"

	"go-embed/pkg/engine"
)

// CalibratedCosineSim computes calibrated similarity adjusting for E5 isotropic bias (~0.70).
func CalibratedCosineSim(raw float32) float32 {
	const baseline = 0.70
	if raw <= baseline {
		return 0
	}
	return (raw - baseline) / (1.0 - baseline)
}

func TestSemanticSanityAndCrossLingual(t *testing.T) {
	eng := loadTestModel(t)

	t.Run("Cross-Lingual Equivalence", func(t *testing.T) {
		// Set 1: Distributed consensus
		queriesConsensus := []struct {
			lang string
			text string
		}{
			{"en", "query: How to implement consensus in distributed systems?"},
			{"ru", "query: Как реализовать консенсус в распределенных системах?"},
			{"id", "query: Bagaimana cara mengimplementasikan konsensus dalam sistem terdistribusi?"},
			{"de", "query: Wie implementiert man einen Konsens in verteilten Systemen?"},
			{"zh", "query: 如何在分布式系统中实现共识？"},
		}

		if isCI() {
			queriesConsensus = queriesConsensus[:2]
		}

		embsConsensus := make([][]float32, len(queriesConsensus))
		for i, q := range queriesConsensus {
			embs, err := eng.Embed(q.text)
			if err != nil {
				t.Fatalf("Embed %s failed: %v", q.lang, err)
			}
			embsConsensus[i] = embs[0]
		}

		for i := 0; i < len(queriesConsensus); i++ {
			for j := i + 1; j < len(queriesConsensus); j++ {
				sim := engine.CosineSimilarity(embsConsensus[i], embsConsensus[j])
				t.Logf("Cross-Lingual Consensus [%s <-> %s]: CosineSim = %.4f (Calibrated = %.4f)", queriesConsensus[i].lang, queriesConsensus[j].lang, sim, CalibratedCosineSim(sim))
				if sim < 0.80 {
					t.Errorf("Cross-Lingual similarity [%s <-> %s] too low: %.4f (expected >= 0.80)", queriesConsensus[i].lang, queriesConsensus[j].lang, sim)
				}
			}
		}

		// Set 2: Cat sleeping on couch
		queriesCat := []struct {
			lang string
			text string
		}{
			{"en", "query: The cat is sleeping on the couch."},
			{"ru", "query: Кот спит на диване."},
			{"id", "query: Kucing itu sedang tidur di sofa."},
			{"de", "query: Die Katze schläft auf der Couch."},
			{"zh", "query: 猫在沙发上睡觉。"},
		}

		if isCI() {
			queriesCat = queriesCat[:2]
		}

		embsCat := make([][]float32, len(queriesCat))
		for i, q := range queriesCat {
			embs, err := eng.Embed(q.text)
			if err != nil {
				t.Fatalf("Embed %s failed: %v", q.lang, err)
			}
			embsCat[i] = embs[0]
		}

		for i := 0; i < len(queriesCat); i++ {
			for j := i + 1; j < len(queriesCat); j++ {
				sim := engine.CosineSimilarity(embsCat[i], embsCat[j])
				t.Logf("Cross-Lingual Cat [%s <-> %s]: CosineSim = %.4f (Calibrated = %.4f)", queriesCat[i].lang, queriesCat[j].lang, sim, CalibratedCosineSim(sim))
				if sim < 0.80 {
					t.Errorf("Cross-Lingual similarity [%s <-> %s] too low: %.4f (expected >= 0.80)", queriesCat[i].lang, queriesCat[j].lang, sim)
				}
			}
		}
	})

	t.Run("Unrelated Negative Pairs", func(t *testing.T) {
		pairs := []struct {
			name string
			q    string
			neg  string
		}{
			{
				name: "K8s vs Tiramisu",
				q:    "query: Kubernetes cluster deployment on bare metal",
				neg:  "passage: Authentic Italian tiramisu recipe with mascarpone, espresso, and ladyfingers.",
			},
			{
				name: "Matching Engine vs Renaissance Art",
				q:    "query: Low latency order matching engine",
				neg:  "passage: History of Renaissance painting and sculpture in Florence during 15th century.",
			},
		}

		for _, p := range pairs {
			embsQ, err := eng.Embed(p.q)
			if err != nil {
				t.Fatalf("Embed query failed: %v", err)
			}
			embsNeg, err := eng.Embed(p.neg)
			if err != nil {
				t.Fatalf("Embed negative failed: %v", err)
			}

			sim := engine.CosineSimilarity(embsQ[0], embsNeg[0])
			calSim := CalibratedCosineSim(sim)
			t.Logf("Negative pair [%s]: Raw CosineSim = %.4f, Calibrated CosineSim = %.4f", p.name, sim, calSim)

			if sim > 0.78 {
				t.Errorf("Negative pair [%s] raw similarity too high: %.4f (expected <= 0.78)", p.name, sim)
			}
			if calSim > 0.35 {
				t.Errorf("Negative pair [%s] calibrated similarity too high: %.4f (expected <= 0.35)", p.name, calSim)
			}
		}
	})

	t.Run("Asymmetric Retrieval Matching (E5 Prefix Sensitivity)", func(t *testing.T) {
		query := "query: how to implement consensus in distributed systems?"
		relPassage := "passage: Consensus in distributed systems is the process of agreeing on a data value among multiple nodes or processes (e.g. Raft, Paxos)."
		irrelPassage := "passage: Authentic Italian tiramisu recipe with mascarpone, espresso, and ladyfingers."

		embsQ, err := eng.Embed(query)
		if err != nil {
			t.Fatalf("Embed query failed: %v", err)
		}
		embsRel, err := eng.Embed(relPassage)
		if err != nil {
			t.Fatalf("Embed rel passage failed: %v", err)
		}
		embsIrrel, err := eng.Embed(irrelPassage)
		if err != nil {
			t.Fatalf("Embed irrel passage failed: %v", err)
		}

		simRel := engine.CosineSimilarity(embsQ[0], embsRel[0])
		simIrrel := engine.CosineSimilarity(embsQ[0], embsIrrel[0])
		gap := simRel - simIrrel
		calGap := CalibratedCosineSim(simRel) - CalibratedCosineSim(simIrrel)

		t.Logf("Asymmetric Retrieval: Sim(Rel)=%.4f, Sim(Irrel)=%.4f, Raw Gap=%.4f, Calibrated Gap=%.4f", simRel, simIrrel, gap, calGap)

		if gap < 0.15 {
			t.Errorf("Asymmetric retrieval raw gap too low: %.4f (expected >= 0.15)", gap)
		}
		if calGap < 0.40 {
			t.Errorf("Asymmetric retrieval calibrated gap too low: %.4f (expected >= 0.40)", calGap)
		}
	})

	t.Run("Hard Negatives & Semantic Contrasts", func(t *testing.T) {
		sHealthy := "passage: The primary database node is healthy and accepting writes."
		sCrashed := "passage: The primary database node crashed and lost data."

		embsHealthy, err := eng.Embed(sHealthy)
		if err != nil {
			t.Fatalf("Embed healthy failed: %v", err)
		}
		embsCrashed, err := eng.Embed(sCrashed)
		if err != nil {
			t.Fatalf("Embed crashed failed: %v", err)
		}

		sim := engine.CosineSimilarity(embsHealthy[0], embsCrashed[0])
		t.Logf("Hard Negatives Contrast: CosineSim = %.4f (Calibrated = %.4f)", sim, CalibratedCosineSim(sim))

		// Ensure meaningful separation despite high lexical overlap
		if sim >= 0.95 {
			t.Errorf("Hard negatives are too similar (conflated): %.4f (expected < 0.95)", sim)
		}
	})
}
