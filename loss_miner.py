from pytorch_metric_learning import losses
from pytorch_metric_learning import miners

#standard implementation for losses and miners (for now): with default parameters

def get_loss(loss_name):
    if loss_name == "contrastive_loss":
        return losses.contrastive_loss(pos_margin=0, neg_margin=1)
    if loss_name == "triplet_margin":
        return losses.triplet_margin_loss()
    if loss_name == "multisimilarity":
        return losses.multi_similarity_loss()
    if loss_name == "cosface":
        return losses.cosface_loss()
    if loss_name == "arcface": #requires an optimizer; cosine similarity is the only compatible distance
        return losses.arcface_loss()
    if loss_name == "vicreg":
        return losses.vicreg_loss()

def get_miner(miner_name):
    miner = None
    if miner_name == "triplet_margin":
        miner = miners.triplet_margin_miner()
    if miner_name == "multisimilarity":
        miner = miners.multi_similarity_miner()
    if miner_name == "batch_hard":
        miner = miners.batch_hard_miner()
    return miner
