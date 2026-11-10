from fractions import Fraction


def witness():
    scores = list(range(-3, 5))
    worlds = {"A": [1, 1, 49, 49, 60, 60, 80, 100], "B": [0, 40, 55, 55, 60, 60, 65, 65]}
    rows = []
    for world, integers in worlds.items():
        posterior = [Fraction(v, 100) for v in integers]
        prior = sum(posterior) / 8
        assert prior == Fraction(1, 2)
        assert posterior == sorted(posterior)
        for split in range(9):
            negative_correct = sum(1 - p for p in posterior[:split]) / 8
            positive_correct = sum(posterior[split:]) / 8
            accuracy = (negative_correct / (1 - prior) + positive_correct / prior) / 2
            boundary = "(-infinity,-3]" if split == 0 else "(4,infinity)" if split == 8 else f"({scores[split-1]},{scores[split]}]"
            rows.append({"world": world, "positive_prior": float(prior), "n_score_support_points": 8,
                "n_low_scores_predicted_negative": split, "threshold_interval": boundary,
                "balanced_accuracy": float(accuracy), "balanced_accuracy_exact": str(accuracy)})
    for world, best, boundary in (("A", .75, "(0,1]"), ("B", .65, "(-2,-1]")):
        selected = [r for r in rows if r["world"] == world]
        maximizers = [r for r in selected if r["balanced_accuracy"] == max(x["balanced_accuracy"] for x in selected)]
        assert len(maximizers) == 1 and maximizers[0]["balanced_accuracy"] == best and maximizers[0]["threshold_interval"] == boundary
    return rows
