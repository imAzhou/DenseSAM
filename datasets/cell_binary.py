from .load_dataset import LoadDataset

class CellBinaryDataset(LoadDataset):

       METAINFO = dict(
        classes=('Background','Cell',),
        palette=[[255, 255, 255], [47, 243, 15]],
        thing_classes=('Cell'),
        stuff_classes=('Background')
       )

       def __init__(self,**args) -> None:
              super(CellBinaryDataset, self).__init__(**args)
