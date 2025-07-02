import csv
import os
import pyvips
import re
import scanpy as sc
import pandas as pd
import h5py

def get_csv(choice, file, testset, tsv_file_path, image_file_path, per_patch_out_dir, n_radius=0):
    per_patch_file_path_list = []
    image = pyvips.Image.new_from_file(image_file_path)

    if choice == 5:
        # 编译正则表达式
        pattern = re.compile(r'([0-9]+(?:\.[0-9]+)?)x([0-9]+(?:\.[0-9]+)?)$')
        with open(tsv_file_path, 'r') as csv_file:
            reader = csv.reader(csv_file)
            next(reader)  # 跳过标题行
            for row in reader:
                if row:  # 确保行不为空
                    sample_id = row[0]  # 假设第一列包含样本 ID
                    match = pattern.search(sample_id)
                    x_str, y_str = match.groups()
                    if match:
                        # 提取匹配的数字
                        # 尝试将字符串转换为整数或浮点数
                        try:
                            x = int(x_str)
                            y = int(y_str)
                        except ValueError:
                            x = float(x_str)
                            y = float(y_str)
                            x = round(x)
                            y = round(y)

                        print(f'x={x}, y={y}')
                        pixel_x = float(row[2])
                        pixel_y = float(row[1])
                    else:
                        print(f'No match found for: {sample_id}')
                    # 打开图像文件
                    print('Image Width: ', image.width, ' Height: ', image.height)

                    # 生成单个patch，保存到本地
                    left = pixel_x - radius_per_patch // 2
                    top = pixel_y - radius_per_patch // 2
                    print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')
                    try:
                        # 确保裁剪区域不超出图像边界
                        if left + radius_per_patch > image.width or top + radius_per_patch > image.height:
                            raise ValueError("Crop area is out of image bounds")

                        patch = image.crop(left, top, radius_per_patch, radius_per_patch)
                    except pyvips.error.Error as e:
                        print(f"Error: {e}")
                    per_patch_file_path = f'{per_patch_out_dir}/{file}_{x}_{y}.png'
                    per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])
                    patch.write_to_file(per_patch_file_path)
    else:
        with open(tsv_file_path, 'r') as tsv_file:
            lines = tsv_file.readlines()
            for line in lines[1:]:
                fields = line.strip().split('\t')

                if choice == 3 or choice == 1:
                    print(fields)
                    x = int(fields[0])
                    y = int(fields[1])
                    pixel_x = float(fields[4])
                    pixel_y = float(fields[5])
                else:
                    print(fields)
                    x = int(fields[1])
                    y = int(fields[2])
                    pixel_x = float(fields[3])
                    pixel_y = float(fields[4])
                # 打开图像文件
                print('Image Width: ', image.width, ' Height: ', image.height)

                # 生成单个patch，保存到本地
                left = pixel_x - radius_per_patch // 2
                top = pixel_y - radius_per_patch // 2
                print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')
                try:
                    # 确保裁剪区域不超出图像边界
                    if left + radius_per_patch > image.width or top + radius_per_patch > image.height:
                        raise ValueError("Crop area is out of image bounds")

                    patch = image.crop(left, top, radius_per_patch, radius_per_patch)
                except pyvips.error.Error as e:
                    print(f"Error: {e}")
                per_patch_file_path = f'{per_patch_out_dir}/{file}_{x}_{y}.png'
                per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])
                patch.write_to_file(per_patch_file_path)

    print(len(per_patch_file_path_list))

    # 保存单个patch信息到csv文件
    os.makedirs(f'{testset}/Mouse/{file}', exist_ok=True)
    if choice == 5 or choice == 6:
        csv_file_path = f'{testset}/Mouse/{file}/per_patch.csv'
    else:
        csv_file_path = f'{testset}/{file}/per_patch.csv'

    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个patch的数据已保存到 {csv_file_path}')

    return 0
    
    
radius_per_patch = 224          # 一个patch的大小 (224, 224)
num_neighbors = 1         # 邻域大小，5*5
n_radius = radius_per_patch * num_neighbors     # 领域半径
### 生成image和mask的patch
# choice = input("1.her2st\n2.stnet\n3.skin\n4.visium\n5.STimage-1K4M)
choice = 7
if choice == 1:
    patient = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for i in patient:
        if i == 'A' or i == 'B' or i == 'C' or i == 'D':
            for j in range(6):
                testset = f'data/her2st/ST-imgs/{i}/{i + str(j+1)}'
                outdir = 'data/her2st'
                figname = [file for file in os.listdir(testset) if file.endswith('.jpg')]
                image_file_path = testset + '/' + figname[0]
                tsv_file_path = f'{outdir}/ST-spotfiles/{i + str(j+1)}_selection.tsv'
                file = f'{i + str(j+1)}'
                per_patch_out_dir = f'{outdir}/gen_per_patch/{i}/{i + str(j + 1)}'
                os.makedirs(per_patch_out_dir, exist_ok=True)
                get_csv(1, file, outdir, tsv_file_path, image_file_path, per_patch_out_dir, n_radius)
        else:
            for j in range(3):
                testset = f'data/her2st/ST-imgs/{i}/{i + str(j + 1)}'
                outdir = 'data/her2st'
                figname = [file for file in os.listdir(testset) if file.endswith('.jpg')]
                image_file_path = testset + '/' + figname[0]
                tsv_file_path = f'{outdir}/ST-spotfiles/{i + str(j + 1)}_selection.tsv'
                file = f'{i + str(j + 1)}'
                per_patch_out_dir = f'{outdir}/gen_per_patch/{i}/{i + str(j + 1)}'
                os.makedirs(per_patch_out_dir, exist_ok=True)
                get_csv(1, file, outdir, tsv_file_path, image_file_path, per_patch_out_dir, n_radius)
elif choice == 2:
    testset = 'data/stnet'
    for filename in os.listdir(testset + '/ST-imgs'):
        # 检查文件是否以.tif结尾
        if filename.endswith('.tif'):
            # 移除.tif后缀并打印文件名
            file = filename[:-4]
        tsv_file_path = f'{testset}/ST-spotfiles/{file[3:]}_selection.tsv'
        image_file_path = f'{testset}/ST-imgs/{file}.tif'
        # nuclei_file_path = f'{testset}/ST-nuclei/{file}.jpg'
        # edge_file_path = f'{testset}/ST-edge/{file}.jpg'
        per_patch_out_dir = f'{testset}/gen_per_patch/{file}'
        '''n_patches_out_dir = f'{testset}/gen_n_patches'
        nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
        nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
        edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
        edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
        '''
        os.makedirs(per_patch_out_dir, exist_ok=True)
        '''os.makedirs(n_patches_out_dir, exist_ok=True)
        os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
        os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
        os.makedirs(edge_per_patch_out_dir, exist_ok=True)
        os.makedirs(edge_n_patches_out_dir, exist_ok=True)'''
        get_csv(2, file, testset, tsv_file_path, image_file_path, per_patch_out_dir, n_radius)

elif choice == 3:
    testset = 'data/skin'
    for filename in os.listdir(testset + '/ST-imgs'):
        # 检查文件是否以.tif结尾
        if filename.endswith('.jpg'):
            # 移除.tif后缀并打印文件名
            file = filename[:-4]
        tsv_file_path = f'{testset}/ST-spotfiles/{file[11:]}_selection.tsv'
        image_file_path = f'{testset}/ST-imgs/{file}.jpg'
        #nuclei_file_path = f'{testset}/ST-nuclei/{file}.jpg'
        #edge_file_path = f'{testset}/ST-edge/{file}.jpg'
        per_patch_out_dir = f'{testset}/gen_per_patch/{file}'
        '''n_patches_out_dir = f'{testset}/gen_n_patches'
        nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
        nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
        edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
        edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
        '''
        os.makedirs(per_patch_out_dir, exist_ok=True)
        '''os.makedirs(n_patches_out_dir, exist_ok=True)
        os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
        os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
        os.makedirs(edge_per_patch_out_dir, exist_ok=True)
        os.makedirs(edge_n_patches_out_dir, exist_ok=True)'''
        get_csv(file, testset, tsv_file_path, image_file_path, per_patch_out_dir, n_radius)

elif choice == 5:
    testset = 'data/STimage-1K4M/ST'
    subdir = 'Mouse'
    for filename in os.listdir(testset + '/image'):
        print(filename)
        # 检查文件是否以.tif结尾
        if subdir in filename:
            # 移除.tif后缀并打印文件名
            file = filename[:-4]
            tsv_file_path = f'{testset}/coord/{file}_coord.csv'
            image_file_path = f'{testset}/image/{file}.png'
            per_patch_out_dir = f'{testset}/gen_per_patch/{subdir}/{file}'
            os.makedirs(per_patch_out_dir, exist_ok=True)
            get_csv(5, file, testset, tsv_file_path, image_file_path, per_patch_out_dir, n_radius)
        else:
            print("no PCW in filename")
elif choice == 6:
    testset = 'data/HEST-1k'
    subdir = 'MISC'

    for filename in os.listdir(os.path.join(testset, 'wsis')):
        print(filename)
        # 检查文件是否以.tif结尾
        if subdir in filename:
            # 如果 filename 包含 'INT1.tif', 'INT
            file = filename[:-4]
            # 移除.tif后缀并打印文件名
            path = os.path.join(testset, 'st', file + '.h5ad')
            # 读取.adata文件
            adata = sc.read(path)
            df = pd.DataFrame({
                'barcode': adata.obs_names,  # 假设adata.var_names包含barcode信息
                'x': adata.obs['array_col'],  # array_col存储在adata.obs中
                'y': adata.obs['array_row'],  # array_row存储在adata.obs中
                'pixel_x': adata.obs['pxl_row_in_fullres'],
                'pixel_y': adata.obs['pxl_col_in_fullres']
            })

            image_file_path = os.path.join(testset, 'wsis', file + '.tif')
            per_patch_out_dir = os.path.join(testset, 'gen_per_patch', subdir, file)
            os.makedirs(per_patch_out_dir, exist_ok=True)
            # 读取图像
            image = pyvips.Image.new_from_file(image_file_path)

            # 定义每个 patch 的半径
            radius_per_patch = 224  # 假设半径为 224，你可以根据需要修改这个值
            per_patch_file_path_list = []

            # 遍历 DataFrame
            for index, row in df.iterrows():
                x = int(row['x'])
                y = int(row['y'])
                pixel_x = float(row['pixel_x'])
                pixel_y = float(row['pixel_y'])

                print(f'x={x}, y={y}, pixel_x={pixel_x}, pixel_y={pixel_y}')

                # 计算裁剪区域的左上角坐标
                left = int(pixel_x - radius_per_patch // 2)
                top = int(pixel_y - radius_per_patch // 2)

                print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')

                # 确保裁剪区域不超出图像边界
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if left + radius_per_patch > image.width:
                    left = image.width - radius_per_patch
                if top + radius_per_patch > image.height:
                    top = image.height - radius_per_patch

                # 检查调整后的坐标是否有效
                if left < 0 or top < 0 or left + radius_per_patch > image.width or top + radius_per_patch > image.height:
                    print(
                        f"Adjusted crop area is still out of image bounds: left={left}, top={top}, radius={radius_per_patch}")
                    continue

                # 裁剪图像
                patch = image.crop(left, top, radius_per_patch, radius_per_patch)

                # 定义输出文件路径
                per_patch_file_path = os.path.join(per_patch_out_dir, f'{file}_{x}_{y}.png')
                per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])

                # 保存裁剪的 patch 到文件
                patch.write_to_file(per_patch_file_path)
                print(f'Patch saved to: {per_patch_file_path}')

            # 保存裁剪路径到CSV文件
            os.makedirs(os.path.join(testset, subdir, file), exist_ok=True)
            csv_file_path = os.path.join(testset, subdir, file, 'per_patch.csv')

            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
                for row in per_patch_file_path_list:
                    writer.writerow(row)
            print(f'单个patch的数据已保存到 {csv_file_path}')

        else:
            print("no PCW in filename")

elif choice == 7:
    testset = 'data/HEST-1k'
    subdir = 'TENX'

    for filename in os.listdir(os.path.join(testset, 'wsis')):
        print(filename)
        # 检查文件是否以.tif结尾
        if subdir in filename:
            # 如果 filename 包含 'INT1.tif', 'INT
            file = filename[:-4]
            # 移除.tif后缀并打印文件名
            path = os.path.join(testset, 'patches', file + '.h5')
            try:
                path = os.path.join(testset, 'patches', file + '.h5')
                # 读取H5文件
                with h5py.File(path, 'r') as f:
                    # 读取 bar 和 coord 数据
                    barcodes = f['barcode'][:]
                    coords = f['coords'][:]
                barcodes = barcodes.flatten()
                # 检查 barcodes 的数据类型
                print(barcodes)
                if isinstance(barcodes[0], bytes):
                    # 如果是字节字符串，需要解码
                    barcodes_split = [barcode.decode('utf-8').split('x') for barcode in barcodes]
                else:
                    # 如果已经是字符串，直接 split
                    barcodes_split = [barcode.split('x') for barcode in barcodes]
                x_coords = [float(x) for x, y in barcodes_split]
                y_coords = [float(y) for x, y in barcodes_split]

                # 创建 DataFrame
                df = pd.DataFrame({
                    'x': x_coords,
                    'y': y_coords,
                    'pixel_x': coords[:, 0],
                    'pixel_y': coords[:, 1]
                })
            except ValueError:
                path = os.path.join(testset, 'st', file + '.h5ad')
                # 读取.adata文件
                adata = sc.read(path)
                df = pd.DataFrame({
                    'barcode': adata.obs_names,  # 假设adata.var_names包含barcode信息
                    'x': adata.obs['array_col'],  # array_col存储在adata.obs中
                    'y': adata.obs['array_row'],  # array_row存储在adata.obs中
                    'pixel_x': adata.obs['pxl_row_in_fullres'],
                    'pixel_y': adata.obs['pxl_col_in_fullres']
                })

            image_file_path = os.path.join(testset, 'wsis', file + '.tif')
            per_patch_out_dir = os.path.join(testset, 'gen_per_patch', subdir, file)
            os.makedirs(per_patch_out_dir, exist_ok=True)
            # 读取图像
            image = pyvips.Image.new_from_file(image_file_path)

            # 定义每个 patch 的半径
            radius_per_patch = 224  # 假设半径为 224，你可以根据需要修改这个值
            per_patch_file_path_list = []

            # 遍历 DataFrame
            for index, row in df.iterrows():
                x = int(row['x'])
                y = int(row['y'])
                pixel_x = float(row['pixel_x'])
                pixel_y = float(row['pixel_y'])

                print(f'x={x}, y={y}, pixel_x={pixel_x}, pixel_y={pixel_y}')

                # 计算裁剪区域的左上角坐标
                left = int(pixel_x - radius_per_patch // 2)
                top = int(pixel_y - radius_per_patch // 2)

                print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')

                # 确保裁剪区域不超出图像边界
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0
                if left + radius_per_patch > image.width:
                    left = image.width - radius_per_patch
                if top + radius_per_patch > image.height:
                    top = image.height - radius_per_patch

                # 检查调整后的坐标是否有效
                if left < 0 or top < 0 or left + radius_per_patch > image.width or top + radius_per_patch > image.height:
                    print(
                        f"Adjusted crop area is still out of image bounds: left={left}, top={top}, radius={radius_per_patch}")
                    continue

                # 裁剪图像
                patch = image.crop(left, top, radius_per_patch, radius_per_patch)

                # 定义输出文件路径
                per_patch_file_path = os.path.join(per_patch_out_dir, f'{file}_{x}_{y}.png')
                per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])

                # 保存裁剪的 patch 到文件
                patch.write_to_file(per_patch_file_path)
                print(f'Patch saved to: {per_patch_file_path}')

            # 保存裁剪路径到CSV文件
            os.makedirs(os.path.join(testset, subdir, file), exist_ok=True)
            csv_file_path = os.path.join(testset, subdir, file, 'per_patch.csv')

            with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
                for row in per_patch_file_path_list:
                    writer.writerow(row)
            print(f'单个patch的数据已保存到 {csv_file_path}')

        else:
            print("no PCW in filename")
else:
    dataset_choice = input("1.10x_breast_ff1\n2.10x_breast_ff2\n3.10x_breast_ff3")
    testset = 'data/test/10x_breast_ff' + str(dataset_choice)
    tsv_file_path = f'{testset}/ST-spotfiles/{testset}_selection.tsv'
    image_file_path = f'{testset}/ST-imgs/{testset}.tif'
    nuclei_file_path = f'{testset}/ST-nuclei/{testset}.jpg'
    edge_file_path = f'{testset}/ST-edge/{testset}.tif'
    per_patch_out_dir = f'{testset}/gen_per_patch'
    n_patches_out_dir = f'{testset}/gen_n_patches'
    nuc_per_patch_out_dir = f'{testset}/gen_nuc_per_patch'
    nuc_n_patches_out_dir = f'{testset}/gen_nuc_n_patches'
    edge_per_patch_out_dir = f'{testset}/gen_edge_per_patch'
    edge_n_patches_out_dir = f'{testset}/gen_edge_n_patches'
    os.makedirs(per_patch_out_dir, exist_ok=True)
    os.makedirs(n_patches_out_dir, exist_ok=True)
    os.makedirs(nuc_per_patch_out_dir, exist_ok=True)
    os.makedirs(nuc_n_patches_out_dir, exist_ok=True)
    os.makedirs(edge_per_patch_out_dir, exist_ok=True)
    os.makedirs(edge_n_patches_out_dir, exist_ok=True)


'''def get_csv(n_radius=0, testset, tsv_file_path, image_file_path, nuclei_file_path, edge_file_path, per_patch_out_dir, n_patches_out_dir, nuc_per_patch_out_dir, nuc_n_patches_out_dir,
            edge_per_patch_out_dir, edge_n_patches_out_dir):
    per_patch_file_path_list = []
    n_patches_file_path_list = []
    nuc_per_patch_file_path_list = []
    nuc_n_patches_file_path_list = []
    edge_per_patch_file_path_list = []
    edge_n_patches_file_path_list = []

    image = pyvips.Image.new_from_file(image_file_path)
    nuc = pyvips.Image.new_from_file(nuclei_file_path)
    edge = pyvips.Image.new_from_file(edge_file_path)

    with open(tsv_file_path, 'r') as tsv_file:
        lines = tsv_file.readlines()
        for line in lines[1:]:
            fields = line.strip().split('\t')

            print(fields)
            x = int(fields[3])
            y = int(fields[4])
            pixel_x = int(fields[5])
            pixel_y = int(fields[6])

            # 打开图像文件
            print('Image Width: ', image.width, ' Height: ', image.height)
            print('Nuc Width: ', nuc.width, ' Height: ', nuc.height)
            print('Edge Width: ', edge.width, ' Height: ', edge.height)

            # 生成单个patch，保存到本地
            left = pixel_x - radius_per_patch // 2
            top = pixel_y - radius_per_patch // 2
            print(f'left={left}, top={top}, x={x}, y={y}, radius={radius_per_patch}')
            patch = image.crop(top, left, radius_per_patch, radius_per_patch)
            per_patch_file_path = f'{per_patch_out_dir}/{testset}_{x}_{y}.tif'
            per_patch_file_path_list.append([x, y, pixel_x, pixel_y, per_patch_file_path])
            patch.write_to_file(per_patch_file_path)
            nuc_patch = nuc.crop(top, left, radius_per_patch, radius_per_patch)
            nuc_per_patch_file_path = f'{nuc_per_patch_out_dir}/{testset}_{x}_{y}.jpg'
            nuc_per_patch_file_path_list.append([x, y, pixel_x, pixel_y, nuc_per_patch_file_path])
            nuc_patch.write_to_file(nuc_per_patch_file_path)
            edge_patch = edge.crop(top, left, radius_per_patch, radius_per_patch)
            edge_per_patch_file_path = f'{edge_per_patch_out_dir}/{testset}_{x}_{y}.tif'
            edge_per_patch_file_path_list.append([x, y, pixel_x, pixel_y, edge_per_patch_file_path])
            edge_patch.write_to_file(edge_per_patch_file_path)

            # 生成邻域patch，保存到本地
            left = pixel_x - radius_per_patch * num_neighbors // 2
            top = pixel_y - radius_per_patch * num_neighbors // 2
            print(f'neighbor_left={left}, neighbor_top={top}, x={x}, y={y}, radius={n_radius}')
            n_patches = image.crop(top, left, n_radius, n_radius)
            n_patches_file_path = f'{n_patches_out_dir}/{testset}_n_{x}_{y}.tif'
            n_patches_file_path_list.append([x, y, pixel_x, pixel_y, n_patches_file_path])
            n_patches.write_to_file(n_patches_file_path)
            nuc_n_patches = nuc.crop(top, left, n_radius, n_radius)
            nuc_patches_file_path = f'{nuc_n_patches_out_dir}/{testset}_n_{x}_{y}.jpg'
            nuc_n_patches_file_path_list.append([x, y, pixel_x, pixel_y, nuc_patches_file_path])
            nuc_n_patches.write_to_file(nuc_patches_file_path)
            edge_n_patches = edge.crop(top, left, n_radius, n_radius)
            edge_patches_file_path = f'{edge_n_patches_out_dir}/{testset}_n_{x}_{y}.tif'
            edge_patches_file_path_list.append([x, y, pixel_x, pixel_y, edge_n_patches_file_path])
            edge_patches.write_to_file(edge_patches_file_path)

    print(len(per_patch_file_path_list))
    print(len(nuc_per_patch_file_path_list))
    print(len(n_patches_file_path_list))
    print(len(nuc_n_patches_file_path_list))
    print(len(edge_per_patch_file_path_list))
    print(len(edge_n_patches_file_path_list))

    # 保存单个patch信息到csv文件
    csv_file_path = f'{testset}/per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个patch的数据已保存到 {csv_file_path}')

    # 保存邻域patch信息到csv文件
    csv_file_path = f'{testset}/n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/nuc_per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in nuc_per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个nuc_patch的数据已保存到 {csv_file_path}')

    # 保存邻域patch信息到csv文件
    csv_file_path = f'{testset}/nuc_n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in nuc_n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域nuc_patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/edge_per_patch.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in edge_per_patch_file_path_list:
            writer.writerow(row)
    print(f'单个edge_patch的数据已保存到 {csv_file_path}')

    csv_file_path = f'{testset}/edge_n_patches.csv'
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['x', 'y', 'pixel-x', 'pixel-y', 'path'])
        for row in edge_n_patches_file_path_list:
            writer.writerow(row)
    print(f'邻域edge_patch的数据已保存到 {csv_file_path}')
    return 0'''

