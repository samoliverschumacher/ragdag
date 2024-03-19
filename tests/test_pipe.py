from app.pipe import Process, RAGStage, create_links


class Security(Process):

    stage =  RAGStage.SECURITY

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class RCache(Process):

    stage =  RAGStage.RCACHE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class Rewrite(Process):

    stage =  RAGStage.REWRITE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class Retrieve(Process):

    stage =  RAGStage.RETRIEVE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class Rerank(Process):

    stage =  RAGStage.RERANK

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class Consolidate(Process):

    stage =  RAGStage.CONSOLIDATE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class Generate(Process):

    stage =  RAGStage.GENERATE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


class WCache(Process):

    stage =  RAGStage.WCACHE

    def __init__(self, config: dict = {}) -> None: super().__init__(config)

    def _process(self, text, data, errors, sentinel):
        new_text = f"{text} > {self.stage.name!s}"
        return new_text, data, errors, self.next_stage


# Test behaviours
def test_pipe_0():

    security = Security()
    rcache = RCache()
    rewrite = Rewrite()
    retrieve = Retrieve()
    rerank = Rerank()
    consolidate = Consolidate()
    generate = Generate()
    wcache = WCache()

    pipeline = [security, rcache, rewrite, retrieve, rerank, consolidate, generate, wcache]
    text = 'begin'
    data, err, sentinel = {}, [], RAGStage.SECURITY
    for process in pipeline:
        text, data, err, sentinel = process(text, data, err, sentinel)

    assert text == 'begin > SECURITY > RCACHE > REWRITE > RETRIEVE > RERANK > CONSOLIDATE > GENERATE > WCACHE'
    assert err == []
    assert data == {}
    assert sentinel == RAGStage.END


# Test behaviours
def test_pipe_create_links():
    """Helper to set the next stage when some steps in RAG are not needed"""

    security = Security()
    generate = Generate()
    wcache = WCache()

    pipeline = [security, generate, wcache]
    create_links(pipeline)

    text = 'begin'
    data, err, sentinel = {}, [], RAGStage.SECURITY
    for process in pipeline:
        text, data, err, sentinel = process(text, data, err, sentinel)

    assert text == 'begin > SECURITY > GENERATE > WCACHE'
    assert err == []
    assert data == {}
    assert sentinel == RAGStage.END


def test_pipe_cache_hit():
    """Dynamically set the next stage when cache hit"""

    security = Security()
    rcache = RCache()
    rewrite = Rewrite()
    retrieve = Retrieve()
    rerank = Rerank()
    consolidate = Consolidate()
    generate = Generate()
    wcache = WCache()

    def _p(text, data, errors, sentinel):
        if 'cache hit' in text:
            return text, data, errors, RAGStage.GENERATE
        return text, data, errors, sentinel.increment()

    rcache._process = _p

    text = 'begin with cache hit'
    data, err, sentinel = {}, [], RAGStage.SECURITY
    for process in [security, rcache, rewrite, retrieve, rerank, consolidate, generate, wcache]:
        text, data, err, sentinel = process(text, data, err, sentinel)

    assert text == 'begin with cache hit > SECURITY > GENERATE > WCACHE'
    assert err == []
    assert data == {}
    assert sentinel == RAGStage.END
