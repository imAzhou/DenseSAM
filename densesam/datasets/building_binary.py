from .load_dataset import LoadDataset

class BuildingBinaryDataset(LoadDataset):

       METAINFO = dict(
        classes=('Background','Building',),
        palette=[[255, 255, 255], [244, 251, 4]],
        thing_classes=('Building'),
        stuff_classes=('Background')
       )

       def __init__(self,**args) -> None:
              super(BuildingBinaryDataset, self).__init__(**args)
