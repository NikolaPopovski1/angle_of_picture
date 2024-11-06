import unittest
import numpy as np
from matplotlib.pyplot import imread
import pathlib

TEST_PATH = pathlib.Path(__file__).resolve().parent

from horizont import orientacija_horizonta

class TestHorizont(unittest.TestCase):
    def pozeni_test_za_sliko(self,
                             slika_pot: str,
                             orientacija_ref: float,
                             orientacija_flip_ref: float,
                             ):
        slika = imread(slika_pot)
        orientacija_est = orientacija_horizonta(slika)
        
        orientacija_diff = abs(orientacija_est - orientacija_ref)
        self.assertLessEqual(orientacija_diff, np.pi/8,
                             msg=f'ocenjena orientacija {orientacija_est} se od prave orientacije {orientacija_ref} razlikuje za več kot pi/8')

        slika_flip = slika[::-1, :, :]
        orientacija_flip_est = orientacija_horizonta(slika_flip)
        orientacija_flip_diff = abs(orientacija_flip_est - orientacija_flip_ref)
        self.assertLessEqual(orientacija_flip_diff, np.pi/8, 
                             msg=f'ocenjena orientacija {orientacija_flip_est} se od prave orientacije {orientacija_flip_ref} razlikuje za več kot pi/8 (za zrcaljen primer)')

    def test_orientacija_horizonta_primer_1(self):
        self.pozeni_test_za_sliko(TEST_PATH.joinpath('primer_-0.25_pi_rad.png'), -0.25*np.pi, 0.25*np.pi)

    def test_orientacija_horizonta_primer_2(self):
        self.pozeni_test_za_sliko(TEST_PATH.joinpath('primer_0_pi_rad.png'), 0, 0)

    def test_orientacija_horizonta_primer_3(self):
        self.pozeni_test_za_sliko(TEST_PATH.joinpath('primer_0.25_pi_rad.png'), 0.25*np.pi, -0.25*np.pi)

if __name__ == '__main__':
    unittest.main()