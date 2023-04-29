from pytorch_metric_learning import losses
from pytorch_metric_learning import miners

#standard implementation for losses and miners (for now): with default parameters

def get_loss(loss_name):
    if loss_name == "contrastive_loss":
        return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == "triplet_margin":
        return losses.TripletMarginLoss()
    if loss_name == "multisimilarity":
        return losses.MultiSimilarityLoss()
    if loss_name == "cosface":
        return losses.CosFaceLoss(num_classes = 16, embedding_size = 512)
    if loss_name == "arcface": #requires an optimizer; cosine similarity is the only compatible distance
        return losses.ArcFaceLoss(num_classes = 16, embedding_size = 512)
    if loss_name == "vicreg":
        return losses.VICRegLoss()

def get_miner(miner_name):
    miner = None
    if miner_name == "triplet_margin":
        miner = miners.TripletMarginMiner()
    if miner_name == "multisimilarity":
        miner = miners.TripletMarginMiner()
    if miner_name == "batch_hard":
        miner = miners.BatchHardMiner()
    return miner
