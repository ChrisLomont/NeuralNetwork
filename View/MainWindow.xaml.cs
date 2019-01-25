using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Data;
using System.Windows.Documents;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Navigation;
using System.Windows.Shapes;

namespace NeuralNet
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
                pt.X * 28/150.0,
                pt.Y * 28/150.0
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
