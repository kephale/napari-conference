from napari_conference import conference_widget


def test_conference_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()

    # this time, our widget will be a MagicFactory or FunctionGui instance
    my_widget = conference_widget(viewer)

    assert my_widget is not None
