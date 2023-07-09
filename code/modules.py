from typing import List


class Token:
    def __init__(self, s: str, token_id: int, paper_id: str, sec_idx: int, para_idx: int,
                 sent_id: int, final_span_id: int = -1, related_tokens: List[int] = []):
        self.s = s
        assert isinstance(self.s, str)
        self.token_id = token_id.item() if not isinstance(token_id, int) else token_id
        assert isinstance(self.token_id, int)
        self.paper_id = paper_id
        assert isinstance(self.paper_id, str)
        self.sec_idx = sec_idx
        assert isinstance(self.sec_idx, int)
        self.para_idx = para_idx
        assert isinstance(self.para_idx, int)
        self.sent_id = sent_id
        assert isinstance(self.sent_id, int)
        self.final_span_id = final_span_id
        assert isinstance(self.final_span_id, int)

    def __str__(self):
        return '{' + 'token: {}, token_id: {}, paper_id: {}, sec_id: {}, para_id: {}, \
            sent_id: {}, span_id: {}'.format(self.s,
                                             self.token_id,
                                             self.paper_id,
                                             self.sec_idx,
                                             self.para_idx,
                                             self.sent_id,
                                             self.final_span_id) + '}'

    def __repr__(self):
        return '{' + 'token: {}, token_id: {}, paper_id: {}, sec_id: {}, para_id: {}, \
            sent_id: {}, span_id: {}'.format(self.s,
                                             self.token_id,
                                             self.paper_id,
                                             self.sec_idx,
                                             self.para_idx,
                                             self.sent_id,
                                             self.final_span_id) + '}'


class Span:
    def __init__(self, s: str, subspans: List[str], tokens: List[List[Token]]):
        assert isinstance(s, str)
        assert isinstance(tokens, list), 'type: {}; content: {}'.format(type(tokens), tokens)
        assert isinstance(tokens[0][1][0], Token)
        self.s = s
        self.subspans = subspans
        self.tokens = tokens
        self.short_qas = []
        self.long_qas = []

    def __str__(self):
        return 's: {}; tokens: {}; short_qas: {}'.format(self.s, self.tokens, self.short_qas)

    def __repr__(self):
        return 's: {}; tokens: {}; short_qas: {}'.format(self.s, self.tokens, self.short_qas)
