import argparse
from transformers import AutoTokenizer

from utils import *
from train_utils import *
from model import *


def main(args):
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token

    train_loader = get_data_loader(
        'data/in_domain_train.tsv', args.batch_size, tokenizer)
    val_loader = get_data_loader(
        'data/in_domain_dev.tsv', args.batch_size, tokenizer, shuffle=False)
    
    train_losses, val_losses, train_metrics, val_metrics = [], [], [], []

    if args.mode == "gen":
        model = GPT(args.gpt_variant, is_gen=True).to(args.device)
        model.eval()

        # TODO: You can add your super creative prompt here
        prompt = "My name is Inigo Montoya. You killed my father. Prepare to die. I am vengeance."

        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(args.device)
        output = model.generate(input_ids, max_new_tokens=args.max_new_tokens)
        print("", tokenizer.decode(output[0]), sep="\n")

    elif args.mode == "LoRA":    
        model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        for epoch in range(args.epochs):
            train_loss, train_acc = train(model, model, train_loader, args)
            val_loss, val_acc = evaluate(model, val_loader, args)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_acc)
            val_metrics.append(val_acc)
        # TODO: Also plot the training losses and metrics
        plot_losses(train_losses, val_losses, args)
        plot_metrics(train_metrics, val_metrics, args)

        model.save_trainable_params(args.model_path)
        
    elif args.mode == "distil":
        teacher_model = GPT(args.gpt_variant, LoRA_rank=args.LoRA_rank).to(args.device)
        teacher_model.load_trainable_params(args.model_path)
        teacher_model.eval()

        student_model = DistilRNN(768, 768, 1, 2).to(args.device)  # TODO: Implement the student model class
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        for epoch in range(args.epochs):
            train_loss, train_acc = train(teacher_model, student_model, train_loader, args)
            val_loss, val_acc = evaluate(student_model, val_loader, args)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_acc)
            val_metrics.append(val_acc)
        # HINT: You can use an additional parameter in train function to differentiate LoRA and distillation training, no changes in evaluate function required.
        # raise NotImplementedError
        plot_losses(train_losses, val_losses, args)
        plot_metrics(train_metrics, val_metrics, args)
        print(f"DistilRNN model total number of parameters:{student_model.count_parameters()}")
        print("Saving model")
        student_model.save_trainable_params("models/distil.pth")
        
    elif args.mode == "rnn":
        model = DistilRNN(768, 768, 1, 2).to(args.device)
        # TODO: Implement the training loop (fill the train and evaluate functions in train_utils.py)
        for epoch in range(args.epochs):
            train_loss, train_acc = train(model, model, train_loader, args)
            val_loss, val_acc = evaluate(model, val_loader, args)
            print(f"Epoch {epoch+1} | Train Loss: {train_loss} | Train Acc: {train_acc} | Val Loss: {val_loss} | Val Acc: {val_acc}")
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            train_metrics.append(train_acc)
            val_metrics.append(val_acc)
        # raise NotImplementedError
        plot_losses(train_losses, val_losses, args)
        plot_metrics(train_metrics, val_metrics, args)
    else:
        print("Invalid mode")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Assignment 2")
    parser.add_argument("mode", type=str, choices=["gen", "LoRA", "distil", "rnn"], help="Mode to run the program in")
    parser.add_argument("sr_no", type=int, help="5 digit SR number")
    parser.add_argument("--gpu_id", type=int, default=0, help="GPU ID to use")
    parser.add_argument("--gpt_variant", type=str, default="gpt2", choices=["gpt2", "gpt2-medium"], help="Model to use")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="Maximum number of tokens to generate")
    parser.add_argument("--model_path", type=str, default="models/LoRA.pth", help="Path to save the model")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--LoRA_rank", type=int, default=4, help="Low rank matrix bottleneck")
    parser.add_argument("--T", type=float, default=2.0, help="Temperature for distillation")
    parser.add_argument("--stl_weight", type=float, default=0.75, help="Weight for soft target loss")
    # TODO: Add more arguments as needed
    
    args = parser.parse_args()
    args.device = torch.device(
        "cuda" if torch.cuda.is_available() and args.gpu_id >= 0 else\
        "mps" if torch.backends.mps.is_available() else "cpu")
    
    # print every possible argument
    print(f"Arguments: \n")
    for arg in vars(args):
        value= getattr(args, arg)
        if isinstance(value, torch.device):
            value = str(value)
        print(f"{arg:<20} |  {value:<10} ")
    seed_everything(args.sr_no)

    main(args)
