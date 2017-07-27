
#include "MainWindow.h"
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);

    QApplication::setStyle(QStyleFactory::create("fusion"));
    QApplication::setPalette(QApplication::palette());
	
	QFont font;
	font.setFamily("Tahoma");
	font.setPointSizeF(8.5);
	a.setFont(font);

    MainWindow w;
    w.show();
	
    //w.setGeometry(w.geometry().x() + 40, w.geometry().y() - 120, w.geometry().width(), w.geometry().height());
	
    return a.exec();
}
