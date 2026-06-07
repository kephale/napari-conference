from napari_conference import conference_widget


def test_conference_widget(make_napari_viewer, capsys):
    viewer = make_napari_viewer()

    my_widget = conference_widget()

    assert my_widget is not None
