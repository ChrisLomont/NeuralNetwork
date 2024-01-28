using System.Windows;
using System.Windows.Input;

namespace Lomont.NeuralNet.View
{
    /// <summary>
    /// Interaction logic for MainWindow.xaml
    /// </summary>
    public partial class MainWindow : Window
    {
        public MainWindow()
        {
            InitializeComponent();
        }

        private void OnLoaded(object sender, RoutedEventArgs e)
        {
            (DataContext as ViewModel.ViewModel).Dispatcher = Dispatcher;
        }

        Point ToPoint(Point pt)
        {
            return new Point(
                pt.X * 28 / 150.0,
                pt.Y * 28 / 150.0
                );
        }

        private void UIElement_OnMouseDown(object sender, MouseButtonEventArgs e)
        {

            (DataContext as ViewModel.ViewModel).Mouse(0, ToPoint(e.GetPosition(drawGrid)));

        }

        private void UIElement_OnMouseMove(object sender, MouseEventArgs e)
        {
            (DataContext as ViewModel.ViewModel).Mouse(1, ToPoint(e.GetPosition(drawGrid)));
        }

        private void UIElement_OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            (DataContext as ViewModel.ViewModel).Mouse(2, ToPoint(e.GetPosition(drawGrid)));
        }
    }
}
