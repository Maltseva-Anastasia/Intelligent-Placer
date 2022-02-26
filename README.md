# Intelligent-Placer

На вход алгоритма подаётся фотография одного или нескольких предметов, расположенных на светлой горизонтальной поверхности, и многоугольника, нарисованного чёрным маркером на белом листе бумаги А4. Требуется программно определить, можно ли одновременно расположить данные предметы на плоскости так, чтобы они разметились в многоугольнике, или нет.

*Требования к входным данным:*

Предметы:
- Предметы контрастные по отношению к фону (светлая горизонтальная поверхность);
(? Чтобы количественно оценить это, нужно воспользоваться  величиной контрастности, оптимальное значение которой располагается в диапазоне 0.60–0.95 ?)
- Предметы не перекрывают друг друга;
- Толщина предмета не более 3 см;
- Длина предмета не более 21 см;
- Предметы изготовлены из любого безопасного материала;
- Предмет может присутствовать на нескольких фотографиях.

Многоугольник:
- Многоугольник нарисован ярким чёрным маркером в пределах листа А4;
- Максимальное количество вершин многоугольника равно 6;
- Многоугольник выпуклый.

Камера:
- Камера располагается на высоте 35 см;
- Камера располагается под прямым углом к светлой горизонтальной поверхности.

Изображение:
- Разрешение фотографии – 4032x3024;
- Минимальная степень размытости изображения;
(? оценка с помощью фильтра Лапласа ?)
- На изображении не могут присутствовать посторонние предметы.

Источник света:
- Искусственный источник света (лампа) располагается чуть выше камеры;
- Искусственный источник света (лампа) располагается под прямым углом к светлой горизонтальной поверхности;

*Требования к выходным данным:*

Ответ True (Да), если возможно разместить предметы в многоугольнике, или False (Нет), если это невозможно или изображение не соответствует входным требованиям.
При этом при размещении предметов в многоугольнике нельзя, чтобы они перекрывали друг друга.

Датасет доступен по этой [ссылке](https://drive.google.com/drive/u/0/folders/1v_O4n5cpNdBP9IJUE24Z2sLz4DxSF502).

*Изображения предметов и светлой горизонтальной поверхности:*

![brush](https://user-images.githubusercontent.com/60979130/153772349-d3b93651-989f-416c-b950-20ea3ebe2b92.jpeg)
![concealer](https://user-images.githubusercontent.com/60979130/153772350-ca17ad8d-c621-45d9-9ed1-a6e03e4e9f22.jpeg)
![cream](https://user-images.githubusercontent.com/60979130/153772353-131ee85e-a3fa-4d91-bb01-e637343ffb0e.jpeg)
![hairbrush](https://user-images.githubusercontent.com/60979130/153772354-976867a3-55c5-43ee-8f74-43c8aac59eb1.jpeg)
![ink](https://user-images.githubusercontent.com/60979130/153772359-d54cb532-5b05-432d-ada9-3f6718021308.jpeg)
![lighter](https://user-images.githubusercontent.com/60979130/153772360-9dc44b8a-8fd9-468f-861f-371989ac9164.jpeg)
![lipstick](https://user-images.githubusercontent.com/60979130/153772363-ce469e47-652a-43d0-8567-7426dce9ca05.jpeg)
![pen](https://user-images.githubusercontent.com/60979130/153772366-4bb33c91-4110-42b1-95b8-d1e993c11c4c.jpeg)
![scissors](https://user-images.githubusercontent.com/60979130/153772370-58a836c4-1beb-4b7e-bb56-64611505918e.jpeg)
![student](https://user-images.githubusercontent.com/60979130/153772372-c8f329c3-8a15-428c-9c15-356e25d06d15.jpeg)
![light](https://user-images.githubusercontent.com/60979130/153772410-6efd2467-e7fb-45dc-b4eb-daa70db769fa.jpg)

*Примечательные изображения:*

Предмет помещается в многоугольник - возвращает True.
![18](https://user-images.githubusercontent.com/60979130/153773682-ddd77bf5-70de-4083-83a6-34296566ac6f.jpeg)

Множество предметов помещается в многоугольник - возвращает True.
![7](https://user-images.githubusercontent.com/60979130/153773690-a8ba334d-4565-45f7-8820-ed941cd8105b.jpeg)

Предмет не помещается в многоугольник - возвращает False.
![1](https://user-images.githubusercontent.com/60979130/153773708-5e9c3468-0ba7-4fb1-8bf8-af8743b22519.jpeg)

Многоугольник невыпуклый - возвращает False.
![19](https://user-images.githubusercontent.com/60979130/153773726-a76b3a4f-9fbc-45fa-aee9-4319db160197.jpeg)

Несколько многоугольников - возвращает False.
![20](https://user-images.githubusercontent.com/60979130/153773749-dc4540c3-b623-4e7c-9e26-2dfc248ca7d0.jpeg)

Предметы перекрывают друг друга - возвращает False.
![22](https://user-images.githubusercontent.com/60979130/153773755-4455b0bf-12b0-481c-887c-4203bf1fbd73.jpeg)

Многоульник имеет больше 6 вершин - возвращает False.
![9](https://user-images.githubusercontent.com/60979130/153773795-360f51d4-d605-4a16-8a9e-33e368682047.jpeg)

Нет предметов на фотографии - возвращает False.
![4](https://user-images.githubusercontent.com/60979130/153773822-98912799-3b55-4c41-9510-b854f887084e.jpeg)
