import pytest

import xchainer


def test_no_backprop_mode():
    assert xchainer.is_backprop_required()
    assert xchainer.is_backprop_required('graph1')
    assert xchainer.is_backprop_required('graph2')

    with xchainer.no_backprop_mode():
        assert not xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')
    assert xchainer.is_backprop_required()
    assert xchainer.is_backprop_required('graph1')
    assert xchainer.is_backprop_required('graph2')

    with xchainer.no_backprop_mode('graph1'):
        assert xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert xchainer.is_backprop_required('graph2')
    assert xchainer.is_backprop_required()
    assert xchainer.is_backprop_required('graph1')
    assert xchainer.is_backprop_required('graph2')

    with xchainer.no_backprop_mode(('graph1', 'graph2')):
        assert xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')
    assert xchainer.is_backprop_required()
    assert xchainer.is_backprop_required('graph1')
    assert xchainer.is_backprop_required('graph2')


def test_force_backprop_mode():
    with xchainer.no_backprop_mode():
        assert not xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')

        with xchainer.force_backprop_mode():
            assert xchainer.is_backprop_required()
            assert xchainer.is_backprop_required('graph1')
            assert xchainer.is_backprop_required('graph2')
        assert not xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')

        with xchainer.force_backprop_mode('graph1'):
            assert not xchainer.is_backprop_required()
            assert xchainer.is_backprop_required('graph1')
            assert not xchainer.is_backprop_required('graph2')
        assert not xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')

        with xchainer.force_backprop_mode(('graph1', 'graph2')):
            assert not xchainer.is_backprop_required()
            assert xchainer.is_backprop_required('graph1')
            assert xchainer.is_backprop_required('graph2')
        assert not xchainer.is_backprop_required()
        assert not xchainer.is_backprop_required('graph1')
        assert not xchainer.is_backprop_required('graph2')

    with xchainer.force_backprop_mode():
        assert xchainer.is_backprop_required()
        assert xchainer.is_backprop_required('graph1')
        assert xchainer.is_backprop_required('graph2')


def test_is_backprop_required():
    current_context = xchainer.get_default_context()
    another_context = xchainer.Context()

    with xchainer.no_backprop_mode():
        with xchainer.force_backprop_mode('graph1'):
            assert not xchainer.is_backprop_required()
            assert xchainer.is_backprop_required('graph1')
            assert not xchainer.is_backprop_required('graph2')

            assert not xchainer.is_backprop_required(context=current_context)
            assert xchainer.is_backprop_required('graph1', context=current_context)
            assert not xchainer.is_backprop_required('graph2', context=current_context)

            assert xchainer.is_backprop_required(context=another_context)
            assert xchainer.is_backprop_required('graph1', context=another_context)
            assert xchainer.is_backprop_required('graph2', context=another_context)

    with pytest.raises(TypeError):
        xchainer.is_backprop_required(context='foo')
