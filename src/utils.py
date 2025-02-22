from copy import deepcopy

import rootutils

root = rootutils.setup_root(search_from=__file__, pythonpath=True)

from pathlib import Path

import h5py
import numpy as np
import numpy.lib.recfunctions as rf
from dotmap import DotMap

from src.datamodules.physics import Mom4Vec, delR


def read_geant4_file(file_path: Path, group_name: str, num_events: int | None = None) -> DotMap:
    """Reads in an hdf5 file from the geant4 dataset."""
    with h5py.File(file_path, "r") as f:
        jet = f[group_name]["jet"][:num_events]
        lep = f[group_name]["lep"][:num_events]
        met = f[group_name]["met"][:num_events]
        misc = f[group_name]["misc"][:num_events]
        nu = f[group_name]["neutrinos"][:num_events]
        evt_info = f[group_name]["evt_info"][:num_events]

    # Safety clips: make sure all kinematics are < 1 TeV
    # Kinematics are the first 3 columns of each array
    # Except met which is the first 2, then clip sumet to 5 TeV
    jet[..., :3] = np.clip(jet[..., :3], -1e3, 1e3)
    lep[..., :3] = np.clip(lep[..., :3], -1e3, 1e3)
    nu[..., :3] = np.clip(nu[..., :3], -1e3, 1e3)
    met[..., :2] = np.clip(met[..., :2], -1e3, 1e3)
    met[..., 2] = np.clip(met[..., 2], 0, 5e3)

    # Undo the logsquash preprocessing to get the raw kinematics
    jet[..., 3:5] = np.exp(jet[..., 3:5]) - 1  # (px, py, pz, E, M, b-tag)
    lep[..., 3] = np.exp(lep[..., 3]) - 1  # (px, py, pz, E, charge, flavour)

    # Change the particle entries to 4 vector objects
    jet = Mom4Vec(jet)
    lep = Mom4Vec(lep)
    nu = Mom4Vec(nu)

    return DotMap(jet=jet, lep=lep, met=met, misc=misc, nu=nu, evt_info=evt_info)


def read_dilepton_file(file_path: Path, require_tops: bool = False) -> DotMap:
    """Reads in data from an HDF file, returning the collection of information as a
    DotMap object.
    """
    # Read the delphes table as a dotmap object
    file_data = DotMap()
    with h5py.File(file_path, "r") as f:
        table = f["delphes"]
        for key in table:
            try:
                file_data[key] = rf.structured_to_unstructured(table[key][:])
            except Exception:
                file_data[key] = table[key][:]

            # Neutrinos has superfluous PDGID at the front which must be removed
            # They also don't have energy!
            if key == "neutrinos":
                file_data[key] = file_data[key][..., 1:4]

            # Leptons need to be ordered particle/antiparticle just like neutrino
            if key == "leptons":
                order = np.argsort(file_data[key][..., -2])  # orders by charge
                order = np.expand_dims(order, -1)
                file_data[key] = np.take_along_axis(file_data[key], order, axis=1)

    # Change the particle entries to 4 vector objects
    for key in ["MET", "neutrinos", "leptons", "jets"]:
        if key in list(file_data.keys()):
            file_data[key] = Mom4Vec(file_data[key], is_cartesian=False)
            file_data[key].to_cartesian()

    # Get the pairing between the lepton and the jets
    lep_jet, antilep_jet = get_lj_pairing(
        file_data.leptons, file_data.jets, is_b=file_data.jets.oth
    )
    file_data.lep_jet = lep_jet
    file_data.antilep_jet = antilep_jet

    # Count the number of b quarks matched to the jets in the data
    has_b = (file_data.jets_indices == 0).sum(axis=-1).astype("bool")
    has_bbar = (file_data.jets_indices == 1).sum(axis=-1).astype("bool")
    file_data.has_both_bs = has_b & has_bbar

    # Count the number of btagged jets
    file_data.at_least_two_bjets = file_data.jets.oth.sum(axis=(-1, -2)) >= 2

    return file_data


def argmin_last_N_axes(A: np.ndarray, N: np.ndarray) -> np.ndarray:
    """Returns the indices of the minimum values of the last N axes of A."""
    s = A.shape
    new_shp = s[:-N] + (np.prod(s[-N:]),)
    max_idx = A.reshape(new_shp).argmin(-1)
    return np.array(np.unravel_index(max_idx, s[-N:])).transpose()


def get_lj_pairing(leptons: Mom4Vec, jets: Mom4Vec, is_b: Mom4Vec) -> np.ndarray:
    # Calculate all possible lj delta R values
    leptons = deepcopy(leptons)
    jets = deepcopy(jets)
    leptons.mom = np.expand_dims(leptons.mom, 1)
    jets.mom = np.expand_dims(jets.mom, 2)
    del_r = delR(leptons, jets)

    # Make it such that if a jet is not b_tagged it receives a large distance
    del_r = np.squeeze(np.where(np.expand_dims(is_b, 2), del_r, 99999))

    # Get the minimum of the matrix
    min_1 = argmin_last_N_axes(del_r, 2)

    # Mask the matrix such that the same minimum can't be chosen again
    idx = np.arange(len(del_r))
    del_r[idx, min_1[:, 0]] = 99999
    del_r[idx, :, min_1[:, 1]] = 99999

    # Get the minimum again
    min_2 = argmin_last_N_axes(del_r, 2)

    # Typically we want the idx of each jet per input lepton
    lep_jet = np.where(min_1[:, 1] == 0, min_1[:, 0], min_2[:, 1])
    antilep_jet = np.where(min_1[:, 1] == 1, min_1[:, 0], min_2[:, 1])

    # Return them both
    return lep_jet, antilep_jet
