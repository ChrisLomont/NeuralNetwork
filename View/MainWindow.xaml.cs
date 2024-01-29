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

        void OnLoaded(object sender, RoutedEventArgs e)
        {
            if (DataContext is ViewModel.MainViewModel vm)
                vm.Dispatcher = Dispatcher;
        }

        Point ToPoint(Point pt)
        {
            return new Point(
                pt.X * 28 / 150.0,
                pt.Y * 28 / 150.0
                );
        }

        void UIElement_OnMouseDown(object sender, MouseButtonEventArgs e)
        {

            if (DataContext is ViewModel.MainViewModel vm)
                vm.Mouse(0, ToPoint(e.GetPosition(drawGrid)));

        }

        void UIElement_OnMouseMove(object sender, MouseEventArgs e)
        {
            if (DataContext is ViewModel.MainViewModel vm) 
                vm.Mouse(1, ToPoint(e.GetPosition(drawGrid)));
        }

        void UIElement_OnMouseUp(object sender, MouseButtonEventArgs e)
        {
            if (DataContext is ViewModel.MainViewModel vm)
                vm.Mouse(2, ToPoint(e.GetPosition(drawGrid)));
        }
    }
}
