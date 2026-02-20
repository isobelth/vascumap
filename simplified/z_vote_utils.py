import numpy as np

def select_single_vote_and_down(votes: dict, min_span_um: float, z_step_um: float, max_z: int | None = None) -> dict | None:
	if not isinstance(votes, dict) or len(votes) == 0:
		raise ValueError("votes must be a non-empty dict of z -> vote_count.")

	keys = list(votes.keys())
	all_int_keys = all(isinstance(key, (int, np.integer)) for key in keys)
	all_str_keys = all(isinstance(key, str) for key in keys)
	zprefixed_str_keys = all_str_keys and all(str(key).strip().lower().startswith("z") for key in keys)

	if zprefixed_str_keys:
		key_formatter = lambda i: f"z{int(i)}"
	elif all_int_keys:
		key_formatter = lambda i: int(i)
	else:
		key_formatter = lambda i: int(i)

	vote_map: dict[int, int] = {}
	for key, value in votes.items():
		if isinstance(key, (int, np.integer)):
			zi = int(key)
		else:
			txt = str(key).strip().lower()
			if txt.startswith("z"):
				txt = txt[1:]
			zi = int(txt)
		vote_map[zi] = int(value)

	active = sorted([zi for zi, count in vote_map.items() if count >= 1])
	if len(active) == 0:
		return None

	spans: list[tuple[int, int]] = []
	start = active[0]
	prev = active[0]
	for zi in active[1:]:
		if zi == prev + 1:
			prev = zi
			continue
		spans.append((start, prev))
		start = zi
		prev = zi
	spans.append((start, prev))

	def span_score(span: tuple[int, int]):
		s0, s1 = span
		length = s1 - s0 + 1
		vote_sum = sum(vote_map.get(i, 0) for i in range(s0, s1 + 1))
		return (length, vote_sum, -s0)

	seed = max(spans, key=span_score)

	z_step = float(z_step_um)
	if not np.isfinite(z_step) or z_step <= 0:
		raise ValueError("z_step_um must be positive (um).")

	target_layers = max(1, int(np.ceil(float(min_span_um) / z_step)))
	z_min, z_max = int(seed[0]), int(seed[1])

	upper_bound = int(max(vote_map.keys())) if max_z is None else int(max(max_z, max(vote_map.keys())))

	while (z_max - z_min + 1) < target_layers:
		if z_min > 0:
			z_min -= 1
		elif z_max < upper_bound:
			z_max += 1
		else:
			break

	return {key_formatter(i): int(vote_map.get(i, 0)) for i in range(int(z_min), int(z_max) + 1)}
