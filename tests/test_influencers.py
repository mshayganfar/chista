from influencers import Influencers
import pytest


@pytest.fixture(scope="class")
def influencers():
    return Influencers(42)


@pytest.mark.parametrize("filenames, number_of_files", [
    (["file01.jpg", "file02.jpg", "file03.jpg", "file04.jpg", "file05.jpg"], 3),
])
class TestInfluencers():
    def test_get_influencr_random_subsampled_files(self, influencers, filenames, number_of_files):
        subsampled_filename_list = influencers.get_influencr_random_subsampled_files(
            filenames, number_of_files)
        assert len(subsampled_filename_list) == 3
