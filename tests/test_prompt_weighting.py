import unittest

from ..mlx_sd.prompt_weighting import (
    parse_prompt_weights,
    tokenize_with_weights,
    tokenize_with_weights_chunks,
)


class ParsePromptWeightsTests(unittest.TestCase):
    def test_plain_text_has_no_groups(self):
        self.assertEqual(
            parse_prompt_weights("a photo of a cat"),
            [("a photo of a cat", 1.0)],
        )

    def test_explicit_weight(self):
        self.assertEqual(
            parse_prompt_weights("(at the beach:0.9)"),
            [("at the beach", 0.9)],
        )

    def test_explicit_weight_with_surrounding_text(self):
        self.assertEqual(
            parse_prompt_weights("a photo (at the beach:0.9) sunny"),
            [("a photo ", 1.0), ("at the beach", 0.9), (" sunny", 1.0)],
        )

    def test_multiple_weighted_groups(self):
        self.assertEqual(
            parse_prompt_weights("(at the beach:0.9), (in the forest:0.9)"),
            [("at the beach", 0.9), (", ", 1.0), ("in the forest", 0.9)],
        )

    def test_implicit_parens_upweight(self):
        self.assertEqual(parse_prompt_weights("(forest)"), [("forest", 1.1)])

    def test_nested_parens_compound(self):
        chunks = parse_prompt_weights("((forest))")
        self.assertEqual(len(chunks), 1)
        text, weight = chunks[0]
        self.assertEqual(text, "forest")
        self.assertAlmostEqual(weight, 1.1 * 1.1)

    def test_brackets_are_literal_text(self):
        # ComfyUI's default parser only treats () as emphasis syntax; [] is ordinary text.
        self.assertEqual(parse_prompt_weights("[beach]"), [("[beach]", 1.0)])

    def test_escaped_parens_are_literal(self):
        self.assertEqual(
            parse_prompt_weights("a smiley \\(:\\) face"),
            [("a smiley (:) face", 1.0)],
        )

    def test_comma_list_with_every_item_weighted_splits_per_item(self):
        chunks = parse_prompt_weights("(midnight skin:0.9, black skin:0.9, olive skin:0.9)")
        self.assertEqual(
            chunks,
            [
                ("midnight skin", 0.9),
                (", ", 1.0),
                ("black skin", 0.9),
                (", ", 1.0),
                ("olive skin", 0.9),
            ],
        )

    def test_comma_list_with_one_item_unweighted_keeps_old_behavior(self):
        # Only the last item has an explicit weight, so this isn't an unambiguous
        # per-item list -- fall back to ComfyUI's own single-group parsing (the trailing
        # colon owns everything before it in the group, including the earlier commas).
        chunks = parse_prompt_weights("(a, b:0.9)")
        self.assertEqual(chunks, [("a, b", 0.9)])

    def test_plain_comma_list_without_weights_upweights_as_one_group(self):
        chunks = parse_prompt_weights("(red hair, blue eyes)")
        self.assertEqual(chunks, [("red hair, blue eyes", 1.1)])

    def test_nested_group_mixed_with_explicit_weight_falls_back(self):
        # Not every top-level segment has its own explicit weight (the nested "(b)" group
        # doesn't), so this isn't treated as a per-item list -- it falls back to ComfyUI's
        # ordinary single-group parsing, which then reparses "(b)" as its own nested group.
        chunks = parse_prompt_weights("(a:0.9, (b))")
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0], ("a:0.9, ", 1.1))
        text, weight = chunks[1]
        self.assertEqual(text, "b")
        self.assertAlmostEqual(weight, 1.1 * 1.1)

    def test_unbalanced_parens_falls_back_to_plain_text(self):
        # No group ever closes, so every chunk keeps weight 1.0 -- the "(" is preserved as a
        # literal character, just split across chunks at the point it was encountered.
        chunks = parse_prompt_weights("a cat (walking")
        self.assertTrue(all(weight == 1.0 for _text, weight in chunks))
        self.assertEqual("".join(text for text, _weight in chunks), "a cat (walking")


class _FakeTokenizer:
    """Minimal stand-in for a transformers CLIPTokenizer: one token id per word."""

    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 2

    def encode(self, text, add_special_tokens=False):
        words = text.split()
        return [100 + i for i in range(len(words))]


class TokenizeWithWeightsTests(unittest.TestCase):
    def test_shapes_and_padding(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "a photo of a cat", max_length=10)
        self.assertEqual(len(ids), 10)
        self.assertEqual(len(weights), 10)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        # 5 words -> bos + 5 + eos = 7 real tokens, 3 pad tokens
        self.assertEqual(ids[6], tokenizer.eos_token_id)
        self.assertEqual(ids[7:], [tokenizer.pad_token_id] * 3)
        self.assertEqual(weights, [1.0] * 10)

    def test_weighted_segment_only_affects_its_own_tokens(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(
            tokenizer, "a photo (at the beach:0.9) sunny", max_length=12
        )
        # bos, "a", "photo", "at", "the", "beach", "sunny", eos, pad...
        self.assertEqual(weights[0], 1.0)  # bos
        self.assertEqual(weights[1], 1.0)  # "a"
        self.assertEqual(weights[2], 1.0)  # "photo"
        self.assertEqual(weights[3], 0.9)  # "at"
        self.assertEqual(weights[4], 0.9)  # "the"
        self.assertEqual(weights[5], 0.9)  # "beach"
        self.assertEqual(weights[6], 1.0)  # "sunny"
        self.assertEqual(weights[7], 1.0)  # eos

    def test_empty_prompt_is_all_ones(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "", max_length=8)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[1], tokenizer.eos_token_id)
        self.assertEqual(ids[2:], [tokenizer.pad_token_id] * 6)
        self.assertEqual(weights, [1.0] * 8)

    def test_truncates_body_to_leave_room_for_bos_eos(self):
        tokenizer = _FakeTokenizer()
        ids, weights = tokenize_with_weights(tokenizer, "one two three four five", max_length=5)
        self.assertEqual(len(ids), 5)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[-1], tokenizer.eos_token_id)


class TokenizeWithWeightsChunksTests(unittest.TestCase):
    def test_short_prompt_matches_single_chunk_tokenize_with_weights(self):
        tokenizer = _FakeTokenizer()
        chunks = tokenize_with_weights_chunks(tokenizer, "a photo of a cat", chunk_size=8)
        self.assertEqual(len(chunks), 1)
        expected = tokenize_with_weights(tokenizer, "a photo of a cat", max_length=10)
        self.assertEqual(chunks[0], expected)

    def test_empty_prompt_is_one_all_pad_chunk(self):
        tokenizer = _FakeTokenizer()
        chunks = tokenize_with_weights_chunks(tokenizer, "", chunk_size=8)
        self.assertEqual(len(chunks), 1)
        ids, weights = chunks[0]
        self.assertEqual(len(ids), 10)
        self.assertEqual(ids[0], tokenizer.bos_token_id)
        self.assertEqual(ids[1], tokenizer.eos_token_id)
        self.assertEqual(ids[2:], [tokenizer.pad_token_id] * 8)
        self.assertEqual(weights, [1.0] * 10)

    def test_long_prompt_splits_into_multiple_full_chunks(self):
        tokenizer = _FakeTokenizer()
        # 10 words, chunk_size=4 -> chunks of 4, 4, 2 real tokens.
        text = "one two three four five six seven eight nine ten"
        chunks = tokenize_with_weights_chunks(tokenizer, text, chunk_size=4)
        self.assertEqual(len(chunks), 3)
        for ids, weights in chunks:
            self.assertEqual(len(ids), 6)
            self.assertEqual(len(weights), 6)
            self.assertEqual(ids[0], tokenizer.bos_token_id)

        # First two chunks are fully packed (no padding); no content is dropped.
        first_ids, _ = chunks[0]
        second_ids, _ = chunks[1]
        third_ids, _ = chunks[2]
        self.assertNotIn(tokenizer.pad_token_id, first_ids[1:-1])
        self.assertNotIn(tokenizer.pad_token_id, second_ids[1:-1])
        # Last chunk holds the remaining 2 real tokens, then eos, then padding.
        self.assertEqual(third_ids[1:3], [108, 109])
        self.assertEqual(third_ids[3], tokenizer.eos_token_id)
        self.assertEqual(third_ids[4:], [tokenizer.pad_token_id] * 2)

    def test_no_content_is_lost_across_chunks(self):
        tokenizer = _FakeTokenizer()
        text = " ".join(f"word{i}" for i in range(20))
        chunks = tokenize_with_weights_chunks(tokenizer, text, chunk_size=6)
        # Reassemble every real (non bos/eos/pad) token across all chunks, in order.
        special = {tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id}
        recovered = [tid for ids, _w in chunks for tid in ids if tid not in special]
        self.assertEqual(recovered, list(range(100, 120)))

    def test_weights_travel_with_their_tokens_across_chunk_boundary(self):
        tokenizer = _FakeTokenizer()
        # 5 plain words then one weighted word, chunk_size=5 puts the weighted word alone
        # in the second chunk.
        text = "a b c d e (weighted:0.7)"
        chunks = tokenize_with_weights_chunks(tokenizer, text, chunk_size=5)
        self.assertEqual(len(chunks), 2)
        _ids0, weights0 = chunks[0]
        self.assertEqual(weights0[1:6], [1.0] * 5)
        ids1, weights1 = chunks[1]
        self.assertEqual(weights1[1], 0.7)


if __name__ == "__main__":
    unittest.main()
