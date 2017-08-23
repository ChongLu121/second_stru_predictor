import os
import argparse
from Bio.PDB import *
import pandas as pd
import numpy as np
from Bio.PDB.Vector import calc_dihedral, calc_angle
from scipy.optimize import fsolve
from math import cos, pi
import warnings

warnings.filterwarnings('ignore', 'WARNING: Chain')
warnings.filterwarnings('ignore', 'The iteration is not making good progress')
warnings.filterwarnings('ignore', 'Could not assign element ')


class Generator:
    def __init__(self, filepath, df_name):
        self.path = filepath
        self.df_name = df_name
        self.train_df = pd.DataFrame(
            pd.DataFrame(columns=['id', 'residue', 'sheet_score', 'helix_score', 'tau_angle', 'theta_angle', 'subset',
                                  '(i, i+3)', '(i, i+3)r', '(i, i+4)', '(i, i+4)r', '(i, i+5)', '(i, i+5)r']))

    def remove_small_molecule(self, structure):
        """ remove all the water and other molecules in the structure
        and return a list of residues"""
        self.res_lst = []
        aa = ('GLU', 'ALA', 'LEU', 'HIS', 'MET', 'GLN', 'TRP', 'VAL', 'PHE', 'LYS',
              'ILE', 'ASP', 'THR', 'SER', 'ARG', 'CYS', 'ASN', 'TYR', 'PRO', 'GLY')

        res_lst = Selection.unfold_entities(structure, 'R')
        # get the list of all residues
        for residue in res_lst:
            name = residue.get_resname().strip().upper()
            if name in aa:
                self.res_lst.append(residue)

    def extra_second_stru(self, file):
        """read pdb file and return a dataframe of secondary structure elements with ids of residues as index.
        label_dic = {0: none, 1: alpha-helix, 2: pi-helix, 3: 310-helix, 4:parallel-sheet, 5: antiparallel-sheet}
        and treat first sheet as paralle beta sheet"""
        self.sec_dic_df = {}
        with open(file, 'r') as infile:
            helix_dic = {'1': 'alpha-helix', '3': 'pi-helix', '5': '310-helix'}
            sheet_dic = {'0': 'parallel-sheet', '1': 'parallel-sheet', '-1': 'antiparallel-sheet',
                         # strange values in some files
                         '': 'parallel-sheet', '-5': 'antiparallel-sheet', '13': 'antiparallel-sheet'}
            for line in infile:
                if line.startswith('HELIX'):
                    # get ids of residues and the type of helix
                    if line[38:40].strip() in helix_dic:
                        self.sec_dic_df.update({i: helix_dic[line[38:40].strip()]
                                                for i in
                                                range(int(line[21:25].strip()), int(line[33:37].strip()) + 1)})
                elif line.startswith('SHEET'):
                    # get ids of residues and the type of sheet
                    self.sec_dic_df.update({i: sheet_dic[line[38:40].strip()]
                                            for i in range(int(line[22:26].strip()), int(line[33:37].strip()) + 1)})

    def cal_seq_score(self):
        """return a data frame contains score for each residue
        of occurring in helix or sheet"""
        name_lst = []
        # residues' name list
        for res in self.res_lst:
            name = res.get_resname().strip().upper()
            name_lst.append(name)
        # score dictionary
        p_helix = {'GLU': 1.53, 'ALA': 1.45, 'LEU': 1.34, 'HIS': 1.24, 'MET': 1.20,
                   'GLN': 1.17, 'TRP': 1.14, 'VAL': 1.14, 'PHE': 1.12, 'LYS': 1.07,
                   'ILE': 1.00, 'ASP': 0.98, 'THR': 0.82, 'SER': 0.79, 'ARG': 0.79,
                   'CYS': 0.77, 'ASN': 0.73, 'TYR': 0.61, 'PRO': 0.59, 'GLY': 0.53}
        p_sheet = {'MET': 1.67, 'VAL': 1.65, 'ILE': 1.60, 'CYS': 1.30, 'TYR': 1.29,
                   'PHE': 1.28, 'GLN': 1.23, 'LEU': 1.22, 'THR': 1.20, 'TRP': 1.19,
                   'ALA': 0.97, 'ARG': 0.90, 'GLY': 0.81, 'ASP': 0.80, 'LYS': 0.74,
                   'SER': 0.72, 'HIS': 0.71, 'ASN': 0.65, 'PRO': 0.62, 'GLU': 0.26}
        # generate two score lists
        helix_lst = [p_helix[res] for res in name_lst]
        sheet_lst = [p_sheet[res] for res in name_lst]
        id_lst = [res.get_id()[1] for res in self.res_lst]
        self.res_df = pd.DataFrame({'id': id_lst, 'residue': name_lst, 'sheet_score': sheet_lst})
        self.res_df['helix_score'] = helix_lst
        return self.res_df

    def get_ca_lst(self):
        """get all C-alpha atoms """
        ca_lst = []
        for residue in self.res_lst:
            for atom in residue:
                if atom.get_name() == 'CA':
                    ca_lst.append(atom)
        return ca_lst

    def cal_torsion_angles(self):
        """calculate tau torsion angles for each 4 consecutive atoms,
        and theta torsion angles for each 3 consecutive atoms"""
        ca_lst = self.get_ca_lst()
        tau_lst = []
        theta_lst = []
        for i in range(len(ca_lst) - 3):
            atoms = [ca_lst[i], ca_lst[i + 1], ca_lst[i + 2], ca_lst[i + 3]]
            vectors = [atom.get_vector() for atom in atoms]
            tau = calc_dihedral(vectors[0], vectors[1], vectors[2], vectors[3])
            tau_lst.append(tau)
        # last three residues
        for k in range(3):
            tau_lst.append(tau_lst[-1])

        for i in range(len(ca_lst) - 2):
            atoms = [ca_lst[i], ca_lst[i + 1], ca_lst[i + 2]]
            vectors = [atom.get_vector() for atom in atoms]
            theta = calc_angle(vectors[0], vectors[1], vectors[2])
            theta_lst.append(theta)
        # last two residues
        for k in range(2):
            theta_lst.append(theta_lst[-1])

        return tau_lst, theta_lst

    def cal_h_coord(self, residu):
        """return coordinate of a hydrogen based on the coordinates of C-alpha, N, O.
        N-H bond length is 0.97A, C-alpha-N-H angle is 119"""
        # get cooridnates of C, N and O atoms
        for atom in residu:
            if atom.get_name() == 'C':
                c = atom.coord
            if atom.get_name() == 'N':
                n = atom.coord
            if atom.get_name() == 'O':
                o = atom.coord

        # equation of plane O=C-N-H
        p1 = np.mat('{}, {}, {}; {}, {}, {}; {}, {}, {}'
                    .format(c[0], c[1], c[2], n[0], n[1], n[2], o[0], o[1], o[2]))
        p2 = np.mat('100, 100, 100').T
        p = np.linalg.solve(p1, p2)

        ang = cos(119 * pi / 180)
        cn = c - n

        def func(i):
            x, y, z = i.tolist()
            return [
                # bond length
                (x - n[0]) ** 2 + (y - n[1]) ** 2 + (z - n[2]) ** 2,
                # bond angle
                np.abs(cn[0] * (n[0] - x) + cn[1] * (n[1] - y) + cn[2] * (n[2] - z)),
                # planarity of the O=C-N-H
                p.A[0][0] * x + p.A[1][0] * y + p.A[2][0] * z
            ]

        res = fsolve(func, [0.97 ** 2, ang * np.linalg.norm(cn) * 0.97, 100], xtol=1e-06, maxfev=500)

        return res

    def cal_electro_energy(self, acceptor, donor):
        """calculate electrostatic energy for each acceptor/donor pair"""
        for atom in donor:
            if atom.get_name() == 'C':
                cd = atom.coord
            if atom.get_name() == 'O':
                od = atom.coord

        for atom in acceptor:
            if atom.get_name() == 'N':
                na = atom.coord
            if atom.get_name() == 'H-N':
                ha = atom.coord
        r = [np.linalg.norm(i) for i in [od - na, cd - ha, od - ha, cd - na]]
        energy = 0.084 * (1 / r[0] + 1 / r[1] - 1 / r[2] - 1 / r[3]) * 332
        return energy

    def cal_all_energy(self):
        """return data frame of hydrogen bonds"""
        # help function
        self.res = None

        def energy_p(n, r):
            if r:
                en = [self.cal_electro_energy(self.res_lst[i], self.res_lst[i + n])
                      for i in range(len(self.res_lst) - n)]
            else:
                en = [self.cal_electro_energy(self.res_lst[i + n], self.res_lst[i])
                      for i in range(len(self.res_lst) - n)]
            for i in range(n):
                en.append(en[-1])
            return en

        # (i, i+3)
        en1 = energy_p(3, True)
        # (i, i+4)
        en2 = energy_p(4, True)
        # (i, i+5)
        en3 = energy_p(5, True)
        # (i, i+3)
        en4 = energy_p(3, False)
        # (i, i+4)
        en5 = energy_p(4, False)
        # (i, i+4)
        en6 = energy_p(5, False)

        self.res = pd.DataFrame({'(i, i+3)': en1, '(i, i+4)': en2, '(i, i+5)': en3,
                                 '(i, i+3)r': en4, '(i, i+4)r': en5, '(i, i+5)r': en6})

    def read_files(self):
        """read file path and write an csv file for prediction model"""
        print('Begin generating data frame...')
        parser = PDBParser(PERMISSIVE=1)
        files = os.listdir(self.path)

        for pdb in files:
            restart = False
            df, feature_df = None, None
            res_lst = None
            structure = None
            if not os.path.isdir(pdb) and pdb.endswith('.pdb'):
                filepath = str(self.path + '/' + pdb)

                structure = parser.get_structure(pdb, filepath)
                # generate a residue list
                self.remove_small_molecule(structure)

                # strip the no-protein files
                if len(self.res_lst) < 4:
                    # print(pdb)
                    continue

                tau_lst, theta_lst = self.cal_torsion_angles()
                # drop files with no label on all C-alpha atoms
                if len(self.res_lst) != len(tau_lst):
                    # print(pdb)
                    continue

                # add hydrogen
                for residue in self.res_lst:
                    try:
                        h_coord = self.cal_h_coord(residue)
                    except:
                        restart = True
                    # add estimated hydrogens, and ignore other parameters
                    hydrogen = Atom.Atom('H-N', h_coord, 0, 0, '', 'H-N', 0, 'H')
                    residue.add(hydrogen)
                if restart:
                    # drop files without C, N, O atoms in the structures
                    # print(pdb)
                    continue

                # generate dataframe
                df = self.cal_seq_score()
                df['tau_angle'], df['theta_angle'] = tau_lst, theta_lst
                # get the label of residues
                self.extra_second_stru(self.path + '/' + pdb)
                df['subset'] = df.id.apply(lambda x: self.sec_dic_df[x] if x in self.sec_dic_df else 'none')
                # calculate hydrogen bond pattern parameter
                self.cal_all_energy()
                feature_df = pd.concat([df, self.res], axis=1)

            self.train_df = pd.concat([self.train_df, feature_df])
            # write an csv file
            self.train_df.to_csv(self.df_name + '.csv')
        print('Generator ends. \n File name: {}\n'.format((self.df_name + '.csv')))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="data frame generator")
    parser.add_argument("folder", help="Input folder path includes pdb files")
    parser.add_argument("output_file_name", help="name of output csv file")
    args = parser.parse_args()

    g = Generator(args.folder, args.output_file_name)
    g.read_files()
