from enum import Enum


class ExtendedEnum(Enum):

    @classmethod
    def values(cls):
        return list(map(lambda c: c.value, cls))


class Hybridization(ExtendedEnum):
    S = "S"
    SP = "SP"
    SP2 = "SP2"
    SP3 = "SP3"
    SP3D = "SP3D"
    SP3D2 = "SP3D2"
    OTHER = "OTHER"


class FormalCharge(ExtendedEnum):
    negTHREE = -3
    negTWO = -2
    negONE = -1
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    EXTREME = "Extreme"


class Degree(ExtendedEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    MORE = "MoreThanFour"


class Chiral(ExtendedEnum):
    CHI_UNSPECIFIED = "CHI_UNSPECIFIED"
    CHI_TETRAHEDRAL_CW = "CHI_TETRAHEDRAL_CW"
    CHI_TETRAHEDRAL_CCW = "CHI_TETRAHEDRAL_CCW"
    CHI_OTHER = "CHI_OTHER"


class TotalNumHs(ExtendedEnum):
    ZERO = 0
    ONE = 1
    TWO = 2
    THREE = 3
    FOUR = 4
    MORE = "MoreThanFour"


class Stereo(ExtendedEnum):
    STEREOZ = "STEREOZ"
    STEREOE = "STEREOE"
    STEREOANY = "STEREOANY"
    STEREONONE = "STEREONONE"


class InRing(ExtendedEnum):
    TRUE = True
    FALSE = False


class IsAromatic(ExtendedEnum):
    TRUE = True
    FALSE = False


class IsConjugated(ExtendedEnum):
    TRUE = True
    FALSE = False


class FeatTables(ExtendedEnum):
    ADJACENT = 'adj_mat'
    ATOM = 'atom_feat'
    BOND = 'bond_feat'


class MolType(ExtendedEnum):
    NUCLEOTIDE = 'nucleo'
    LIGAND = 'ligand'
    PROTEIN = 'protein'
    TF = 'transcription_factor'
    PRIMER = 'primer'
    PROMOTER = 'promoter'
    ENHANCER = 'enhancer'
    RNA = 'RNA'
    DNA = 'DNA'


class MolDataStruct(ExtendedEnum):
    SEQUENCE = 'seq'
    SMILE = 'smile'
