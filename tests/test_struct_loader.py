"""Test the StructLoader class."""

import os
from quickviewer.data.structures import StructLoader
from matplotlib.colors import to_rgba


# Load single structure from single file
def test_single():
    s = StructLoader("data/structs/my_structs/cube.nii")
    assert len(s.get_structs()) == 1
    assert not len(s.get_comparisons())
    #  assert not len(s.get_structs(ignore_unpaired=True))
    assert s.get_structs()[0].name == "Cube"

#  # Default colour
#  def test_default_color():
    #  s = StructLoader("data/structs/my_structs/cube.nii",
                     #  names={"*cube*": "right parotid"})
    #  struct = s.get_structs()[0]
    #  assert struct.color == to_rgba("red")

# Custom name and colour
def test_custom_name():
    names = {"custom": "*cube.nii"}
    colors = {"custom": "purple"}
    s = StructLoader("data/structs/my_structs/cube.nii", names=names,
                     colors=colors)
    struct = s.get_structs()[0]
    assert struct.name == "custom"
    assert struct.color == to_rgba("purple")
    assert struct.loaded
    assert struct.label == ""

# Load multiple structures from directory
def test_dir():
    colors = {"cube": "red", "sphere": "green"}
    s = StructLoader(structs="data/structs/my_structs/", colors=colors)
    structs = s.get_structs()
    assert(len(structs) == 3)
    names_colors = {s.name: s.color for s in structs}
    assert names_colors["Cube"] == to_rgba(colors["cube"])
    assert names_colors["Sphere"] == to_rgba(colors["sphere"])

# Load structures using wildcard filename
def test_wildcard():
    names = {"with_cube": "*cube*", "sphere_only": "*"}
    s = StructLoader("data/structs/my_structs/sphere*", names=names)
    snames = [s.name for s in s.get_structs()]
    assert len(snames) == 2
    assert sorted(snames) == sorted(list(names.keys()))

# Load multiple structure masks from one file
def test_multi_structs():
    s = StructLoader(
        multi_structs="data/structs/my_structs/sphere_and_cube.nii")
    structs = s.get_structs()
    assert(len(structs) == 2)
    assert structs[0].name == "Structure 1"

# Test list of structure names inside file
def test_many_names():
    names = ["cube", "sphere"]
    colors = {"cube": "green"}
    s = StructLoader(
        multi_structs="data/structs/my_structs/sphere_and_cube.nii",
        names=names, colors=colors)
    snames = [s.name for s in s.get_structs()]
    assert sorted(snames) == sorted(names)
    assert [s for s in s.get_structs() if s.name == "cube"][0].color == \
            to_rgba("green")

# Load structures from list of files
def test_list():
    s = StructLoader(["data/structs/my_structs", 
                      "data/structs/my_structs/subdir"])
    assert len(s.get_structs()) == 5

# Load structures with labels
def test_labels():

    # Load structures from multiple sources
    multi = {"set1": "data/structs/my_structs/sphere_and_cube.nii"}
    structs = {"set2": "data/structs/my_structs/subdir"}
    names = {"set1": ["Sphere", "Cube"]}
    colors = {"set1": {"*": "green"}, 
              "set2": {"cube": "yellow", "sphere": "black"}}
    s = StructLoader(structs, multi, names=names, colors=colors)

    # Test structure properties
    structs = s.get_structs()
    assert len(structs) == 4
    assert len([s for s in structs if s.label == "set1"]) == 2
    assert len([s for s in structs if s.label == "set2"]) == 2
    assert len([s for s in structs if s.name == "Sphere"]) == 2
    assert len([s for s in structs if s.name == "Cube"]) == 2
    assert [s.color for s in structs if s.label == "set1"][0] == \
            to_rgba("green")
    assert [s.color for s in structs if s.label == "set1"][1] == \
            to_rgba("green")
    assert [s.color for s in structs if s.label == "set2"][1] == \
            to_rgba("yellow")
    assert [s.color for s in structs if s.label == "set2"][0] == \
            to_rgba("black")

    # Test comparisons
    comps = s.get_comparisons()
    assert len(comps) == 2
    assert sorted([c.name for c in comps]) == sorted(["Sphere", "Cube"])
    assert len(s.get_structs(False)) == len(s.get_structs(True))
    assert not len(s.get_standalone_structs())

# Test extraction of standalone structs
def test_standalone():

    # Load structures from multiple sources
    multi = {"set1": "data/structs/my_structs/sphere_and_cube.nii"}
    structs = {"set2": "data/structs/my_structs/subdir"}
    names = {"set1": ["Sphere"]}
    s = StructLoader(structs=structs, multi_structs=multi, names=names)
    assert len(s.get_structs()) == 4
    assert len(s.get_structs(True)) == 2
    assert len(s.get_comparisons()) == 1
    assert len(s.get_standalone_structs()) == 2

# Load pairs of structures for comparison
def test_pairs():
    pairs = [
        ["data/structs/my_structs/sphere.nii", 
         "data/structs/my_structs/cube.nii"],
        ["data/structs/my_structs/sphere_and_cube.nii",
         "data/structs/my_structs/subdir/sphere.nii"]]
    s = StructLoader(pairs)
    assert len(s.get_structs()) == 4
    assert len(s.get_comparisons()) == 2
    assert len(s.get_structs(True)) == 4
    assert len(s.get_standalone_structs()) == 0

# Comparison of two structs
def test_two_comparison():
    s = StructLoader("data/structs/my_structs/sphere*")
    assert not len(s.get_standalone_structs())
    assert len(s.get_structs(True)) == 2
    assert len(s.get_comparisons()) == 1
