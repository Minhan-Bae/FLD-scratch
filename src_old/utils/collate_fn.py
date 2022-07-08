def collate_fn(batch):
    return zip(*batch)