import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_history(history, epoch_count, y_lim_loss=None):
    # Preprocessing
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, epoch_count + 1)
    df_loss = pd.melt(pd.DataFrame(zip(epochs, loss, val_loss), columns=["Epoch", "Train", "Validation"]),
                      id_vars="Epoch",
                      value_vars=["Train", "Validation"],
                      var_name="Dataset",
                      value_name="Loss (Cross entropy)")
    df_acc = pd.melt(pd.DataFrame(zip(epochs, acc, val_acc), columns=["Epoch", "Train", "Validation"]),
                     id_vars="Epoch",
                     value_vars=["Train", "Validation"],
                     var_name="Dataset",
                     value_name="Accuracy")
    df = pd.merge(df_loss, df_acc)

    # Seaborn visualization
    # Configure Seaborn theme
    sns.set_theme(style="darkgrid", palette=sns.color_palette("viridis"))
    # Plot loss
    gl = sns.lineplot(data=df_loss, x="Epoch", y="Loss (Cross entropy)", hue="Dataset")
    gl.set_ylim(0, y_lim_loss)
    plt.show()
    # Plot accuracy
    ga = sns.lineplot(data=df_acc, x="Epoch", y="Accuracy", hue="Dataset")
    ga.set_ylim(0, 1)
    plt.show()
    # Both in one
    g = sns.PairGrid(df, x_vars='Epoch', y_vars=['Loss (Cross entropy)', 'Accuracy'],
                     hue="Dataset")
    g.map(sns.lineplot)
    g.fig.set_size_inches(8, 8)
    g.axes[0, 0].set_ylim(0, y_lim_loss)
    g.axes[1, 0].set_ylim(0, 1)
    g.add_legend()
    # sns.despine(fig=g.fig)
    plt.show()
    # End
