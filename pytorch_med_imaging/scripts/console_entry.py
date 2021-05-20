import argparse
import os


class pmi_console_entry(argparse.ArgumentParser):
    def __init__(self):
        super(pmi_console_entry, self).__init__()

    def make_console_entry_io(self):
        self.add_argument('-i', '--input', type=str, action='store', dest='input',
                          help='Input directory that contains the DICOMs.')
        self.add_argument('-o', '--output', type=str, action='store', dest='output',
                          help='Output directory to hold the nii files.')
        self.add_argument('-g', '--idglobber', action='store', default=None, dest='idglobber',
                          help='Specify the globber to glob the ID from the DICOM paths.')
        self.add_argument('--idlist', default=None,
                          help='Pass ID list to class ImageDataSet.')
        self.add_argument('-v', '--verbose', action='store_true',
                          help="Verbosity.")


    def parse_args(self, *args, **kwargs):
        a = super(pmi_console_entry, self).parse_args(*args, **kwargs)

        # Create output dir
        if hasattr(a, 'output'):
            if not os.path.isdir(a.output):
                os.makedirs(a.output, exist_ok=True)
        return a