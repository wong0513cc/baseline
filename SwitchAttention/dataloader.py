from torch.utils.data import DataLoader
from dataset_v2 import GraphESGDataset

def build_loaders(years_train, years_val, years_test, batch_size, root_paths, **kwargs):
    """回傳 train/val/test 三個 dict，每個 dict 的 key 是年份，value 是 DataLoader"""
    def make_loader(years, shuffle):
        loaders = {}
        for y in years:
            ds = GraphESGDataset(
                root_price=root_paths["price"],
                root_finance=root_paths["finance"],
                root_news=root_paths["news"],
                root_event=root_paths["event"],
                root_graph=root_paths["graph"],
                root_label=root_paths["label"],
                root_year_symbols=root_paths["year_symbols"],
                years=[y],
                has_label=True,
                strict_check=True,
                fill_missing_event="zeros",
                fill_missing_graph="zeros",
                **kwargs
            )
            loaders[y] = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda x: x[0])
        return loaders

    train_loaders = make_loader(years_train, shuffle=True)
    val_loaders   = make_loader(years_val,   shuffle=False)
    test_loaders  = make_loader(years_test,  shuffle=False)

    return train_loaders, val_loaders, test_loaders