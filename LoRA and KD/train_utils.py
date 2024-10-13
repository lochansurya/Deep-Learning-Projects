import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from argparse import Namespace

def train(teacher_model: nn.Module,
          student_model: nn.Module,
          train_loader: DataLoader,
          args: Namespace,):
    # pass
    """
    LoRA: There are two models to consider here for knowledge distillation: teacher and student.
    When LoRA is the mode in the args namespace, the teacher model and the student model will be the same but 
    call only the teacher model for training and evaluation.

    distil: The teacher model is the GPT model and the student model is the DistilRNN model.
    When distil is the mode in the args namespace, the teacher model is the GPT model and the student model is the DistilRNN model.
    The teacher model is in eval mode and the student model is in train mode.

    rnn: Both of the models are student model, the DistilRNN model.
    When rnn is the mode in the args namespace, the student model is the DistilRNN model.
    Call the student model for training and evaluation.

    loss_fn: the criterion (or) the loss function to be used for training the student model as well as the teacher model.
            We use the CrossEntropyLoss function for the loss function.
    
    optimizer: the optimizer to be used for training the student model.
            We use the Adam optimizer for the optimizer. The learning rate is fetched from the args namespace.

    evaluate the loss and accuracy over the training data and return the loss and accuracy.
    """
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(student_model.parameters(),lr=args.lr)
    train_loss=0
    train_acc=0
    if args.mode=="LoRA":
        # train the teacher model
        teacher_model.train()
        for X, mask, y in train_loader:
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = teacher_model(X, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_acc+=torch.sum(torch.argmax(logits, dim=1)==y).item()
        train_acc/=len(train_loader.dataset)
        train_loss/=len(train_loader.dataset)

    elif args.mode=="distil":
        # check if the teacher_model is in eval mode
        if teacher_model.training:
            teacher_model.eval()
        # train the student model
        student_model.train()
        for X, mask, y in train_loader:
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            with torch.no_grad():
                teacher_logits=teacher_model(X, mask)
            student_logits = student_model(X, mask)
            """
            https://pytorch.org/tutorials/beginner/knowledge_distillation_tutorial.html#knowledge-distillation-run
            """
            soft_targets= F.softmax(teacher_logits/ args.T, dim=-1)
            soft_prob= F.log_softmax(student_logits/ args.T, dim=-1)

            soft_targets_loss= torch.sum(soft_targets * (soft_targets.log() - soft_prob)) / soft_prob.size()[0] * (args.T**2)
            label_loss= loss_fn(student_logits, y)
            loss= args.stl_weight * soft_targets_loss + (1-args.stl_weight) * label_loss
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_acc+=torch.sum(torch.argmax(student_logits, dim=1)==y).item() 
        train_acc/=len(train_loader.dataset)
        train_loss/=len(train_loader.dataset)


    elif args.mode=="rnn":
        student_model.train()
        for X, mask, y in train_loader:
            X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
            optimizer.zero_grad()
            logits = student_model(X, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()
            train_loss+=loss.item()
            train_acc+=torch.sum(torch.argmax(logits, dim=1)==y).item() 
        train_acc/=len(train_loader.dataset)
        train_loss/=len(train_loader.dataset)
    
    else :
        print("Invalid mode")
        return
    return train_loss, train_acc



        

def evaluate(model: nn.Module,
             val_loader: DataLoader,
             args: Namespace):
    """
    Evaluate the loss and accuracy for the model over the validation data and return the loss and accuracy.
    Takes the model, the validation data loader and the args namespace as input.
    """
    # pass
    val_loss=0
    val_acc=0
    model.eval()
    loss_fn=nn.CrossEntropyLoss()
    for X, mask, y in val_loader:
        X, mask, y = X.to(args.device), mask.to(args.device), y.to(args.device)
        logits = model(X, mask)
        loss = loss_fn(logits, y)
        val_loss+=loss.item()
        val_acc+=torch.sum(torch.argmax(logits, dim=1)==y).item()
    val_acc/=len(val_loader.dataset)
    val_loss/=len(val_loader.dataset)
    return val_loss, val_acc
