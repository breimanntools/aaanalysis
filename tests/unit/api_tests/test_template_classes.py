"""This is a script to test the Tool and Wrapper template ABCs in aaanalysis.template_classes.

These two abstract base classes define the ``.run``/``.eval`` (Tool) and
``.__init__``/``.fit``/``.eval`` (Wrapper) contracts every public class implements. The
tests instantiate minimal concrete subclasses that call ``super().<method>()`` so the
abstract bodies (the ``raise NotImplementedError`` guards and the Wrapper attribute
initialisation) are actually executed — a subclass that *forgets* to override a method
must still get a clear ``NotImplementedError``, not silent ``None``.
"""
import pytest

from aaanalysis.template_classes import Tool, Wrapper


# Concrete minimal subclasses that defer to the abstract bodies via super().
class _ConcreteTool(Tool):
    def run(self):
        return super().run()

    def eval(self):
        return super().eval()


class _ConcreteWrapper(Wrapper):
    def __init__(self):
        super().__init__()

    def fit(self, *args, **kwargs):
        return super().fit(*args, **kwargs)

    def eval(self, *args, **kwargs):
        return super().eval(*args, **kwargs)


class TestTool:
    """Tool ABC: abstract run/eval must raise NotImplementedError; the ABC is not instantiable."""

    def test_cannot_instantiate_abstract_tool(self):
        with pytest.raises(TypeError):
            Tool()

    def test_run_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ConcreteTool().run()

    def test_eval_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ConcreteTool().eval()


class TestWrapper:
    """Wrapper ABC: __init__ seeds the model attributes; abstract eval must raise."""

    def test_cannot_instantiate_abstract_wrapper(self):
        with pytest.raises(TypeError):
            Wrapper()

    def test_init_sets_model_attributes_to_none(self):
        wrapper = _ConcreteWrapper()
        assert wrapper._model_class is None
        assert wrapper.model is None
        assert wrapper.model_kwargs is None

    def test_fit_default_body_returns_none(self):
        # Wrapper.fit has no executable body beyond its docstring -> returns None.
        assert _ConcreteWrapper().fit() is None

    def test_eval_raises_not_implemented(self):
        with pytest.raises(NotImplementedError):
            _ConcreteWrapper().eval()
