# ProyectoIMT3810
En este repositorio se encuentra la compresión en el formato de matrices Jerárquicas. Considerando para la compresión de bajo rango el método Adaptive Cross Approximation.
La compresión en matrices Jerárquicas permite disminuir considerablemente el costo de almacenamiento de matrices grande que sean completamente densas.
En este repositorio se encuentran funciones para realizar la compresión de una matriz a formato de matriz jerárquica, para realizar la descomposición en formato ACA(Adpative Cross Approximation) y también para realizar la operación Matvec. Todo esto se relaizó con el ejemplo específico de Single Layer Potential para el problema de Laplace, utilizando funciones de base P1.

## Librerías
Se utilizó la librería Numpy y también Bempp-cl. Para la intalación de esta última librería, se puede realizar con el siguiente comando:

´´´bash
npm python setup.py install
´´´

Para otras formas de instalación se puede revisar el siguiente link: [bempp.com/installation.html](bempp.com/installation.html.)

