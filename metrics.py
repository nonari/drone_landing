
def calc_acc(output, target):
    predictions = output.argmax(dim=1, keepdim=True)
    correct = (predictions == target).sum().item()
    accuracy = correct / target.size().numel()

    return accuracy
