import pytest

import xchainer


_devices_data = [
    {'index': 0},
    {'index': 1},
]


@pytest.fixture(params=_devices_data)
def device_data1(request):
    return request.param


@pytest.fixture(params=_devices_data)
def device_data2(request):
    return request.param


@pytest.fixture
def device_instance1(request, device_data1):
    return xchainer.get_global_default_context().get_device('native', device_data1['index'])


@pytest.fixture
def device_instance2(request, device_data2):
    return xchainer.get_global_default_context().get_device('native', device_data2['index'])


@pytest.fixture
def cache_restore_device(request):
    device = xchainer.get_default_device()

    def restore_device():
        xchainer.set_default_device(device)
    request.addfinalizer(restore_device)


def test_creation():
    ctx = xchainer.get_global_default_context()
    backend = ctx.get_backend('native')
    device = backend.get_device(0)
    assert device.name == 'native:0'
    assert device.backend is backend
    assert device.context is ctx
    assert device.index == 0

    device = backend.get_device(1)
    assert device.name == 'native:1'
    assert device.backend is backend
    assert device.context is ctx
    assert device.index == 1


def test_synchronize():
    ctx = xchainer.get_global_default_context()
    device = ctx.get_device('native', 0)
    device.synchronize()


@pytest.mark.usefixtures('cache_restore_device')
def test_default_device(device_instance1):
    device = device_instance1
    xchainer.set_default_device(device)
    assert xchainer.get_default_device() is device


@pytest.mark.usefixtures('cache_restore_device')
def test_default_device_with_name(device_instance1):
    device = device_instance1
    xchainer.set_default_device(device.name)
    assert xchainer.get_default_device() is device


@pytest.mark.usefixtures('cache_restore_device')
def test_eq(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    device1_1 = device1.backend.get_device(device1.index)
    device1_2 = device1.backend.get_device(device1.index)
    device2_1 = device2.backend.get_device(device2.index)

    assert device1_1 == device1_2
    assert device1_1 != device2_1
    assert not (device1_1 != device1_2)
    assert not (device1_1 == device2_1)


@pytest.mark.usefixtures('cache_restore_device')
def test_device_scope(device_instance1, device_instance2):
    if device_instance1 == device_instance2:
        return

    device1 = device_instance1
    device2 = device_instance2

    xchainer.set_default_device(device1)
    with xchainer.device_scope(device2):
        assert xchainer.get_default_device() == device2

    scope = xchainer.device_scope(device2)
    assert xchainer.get_default_device() == device1
    with scope:
        assert xchainer.get_default_device() == device2
    assert xchainer.get_default_device() == device1
