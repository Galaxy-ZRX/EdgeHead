#####################
# Based on https://github.com/hongzhenwang/RRPN-revise
# Licensed under The MIT License
# Author: yanyan, scrin@foxmail.com
#####################
import math

import numba
import numpy as np
from numba import cuda


@numba.jit(nopython=True)
def div_up(m, n):
    return m // n + (m % n > 0)


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def trangle_area(a, b, c):
    return ((a[0] - c[0]) * (b[1] - c[1]) - (a[1] - c[1]) *
            (b[0] - c[0])) / 2.0


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def area(int_pts, num_of_inter):
    area_val = 0.0
    for i in range(num_of_inter - 2):
        area_val += abs(
            trangle_area(int_pts[:2], int_pts[2 * i + 2:2 * i + 4],
                         int_pts[2 * i + 4:2 * i + 6]))
    return area_val


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def sort_vertex_in_convex_polygon(int_pts, num_of_inter):
    if num_of_inter > 0:
        center = cuda.local.array((2,), dtype=numba.float32)
        center[:] = 0.0
        for i in range(num_of_inter):
            center[0] += int_pts[2 * i]
            center[1] += int_pts[2 * i + 1]
        center[0] /= num_of_inter
        center[1] /= num_of_inter
        v = cuda.local.array((2,), dtype=numba.float32)
        vs = cuda.local.array((16,), dtype=numba.float32)
        for i in range(num_of_inter):
            v[0] = int_pts[2 * i] - center[0]
            v[1] = int_pts[2 * i + 1] - center[1]
            d = math.sqrt(v[0] * v[0] + v[1] * v[1])
            if d == 0:
                break

            v[0] = v[0] / d
            v[1] = v[1] / d
            if v[1] < 0:
                v[0] = -2 - v[0]
            vs[i] = v[0]
        j = 0
        temp = 0
        for i in range(1, num_of_inter):
            if vs[i - 1] > vs[i]:
                temp = vs[i]
                tx = int_pts[2 * i]
                ty = int_pts[2 * i + 1]
                j = i
                while j > 0 and vs[j - 1] > temp:
                    vs[j] = vs[j - 1]
                    int_pts[j * 2] = int_pts[j * 2 - 2]
                    int_pts[j * 2 + 1] = int_pts[j * 2 - 1]
                    j -= 1

                vs[j] = temp
                int_pts[j * 2] = tx
                int_pts[j * 2 + 1] = ty


@cuda.jit('(float32[:], int32)', device=True, inline=True)
def filter_same_pts_zrx(int_pts, num_of_inter):
    int_pts_new = cuda.local.array((12,), dtype=numba.float32)
    num_of_inter_new = num_of_inter
    for i in range(12):
        int_pts_new[i] = int_pts[i]
    j = 0
    for i in range(num_of_inter):
        if int_pts[2 * i] == int_pts[2 * i + 2] and int_pts[2 * i + 1] == int_pts[2 * i + 3]:
            num_of_inter_new = num_of_inter_new - 1
            j += 1
        int_pts_new[2 * i] = int_pts[2 * j]
        int_pts_new[2 * i + 1] = int_pts[2 * j + 1]
        j += 1
    sort_vertex_in_convex_polygon(int_pts_new, num_of_inter_new)
    return int_pts_new, num_of_inter_new


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2,), dtype=numba.float32)
    B = cuda.local.array((2,), dtype=numba.float32)
    C = cuda.local.array((2,), dtype=numba.float32)
    D = cuda.local.array((2,), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 4)]
    B[1] = pts1[2 * ((i + 1) % 4) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 4)]
    D[1] = pts2[2 * ((j + 1) % 4) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH

            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_v1(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 4)]
    b[1] = pts1[2 * ((i + 1) % 4) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 4)]
    d[1] = pts2[2 * ((j + 1) % 4) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    return True


@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral(pt_x, pt_y, corners):
    ab0 = corners[2] - corners[0]
    ab1 = corners[3] - corners[1]

    ad0 = corners[6] - corners[0]
    ad1 = corners[7] - corners[1]

    ap0 = pt_x - corners[0]
    ap1 = pt_y - corners[1]

    abab = ab0 * ab0 + ab1 * ab1
    abap = ab0 * ap0 + ab1 * ap1
    adad = ad0 * ad0 + ad1 * ad1
    adap = ad0 * ap0 + ad1 * ap1

    return abab >= abap and abap >= 0 and adad >= adap and adap >= 0

@cuda.jit('(float32, float32, float32[:])', device=True, inline=True)
def point_in_quadrilateral_v2(pt_x, pt_y, corners):
    wn = 0
    on_edge = False
    for i in range(4):
        x1, y1 = corners[2*i], corners[2*i+1]
        x2, y2 = corners[(2*i+2)%8], corners[(2*i+3)%8]
        if y1 <= pt_y:
            if y2 > pt_y and (x2-x1)*(pt_y-y1) > (pt_x-x1)*(y2-y1):
                wn += 1
            elif y2 == pt_y and min(x1, x2) <= pt_x <= max(x1, x2):
                on_edge = True
        else:
            if y2 <= pt_y and (x2-x1)*(pt_y-y1) < (pt_x-x1)*(y2-y1):
                wn -= 1
            elif y2 == pt_y and min(x1, x2) <= pt_x <= max(x1, x2):
                on_edge = True
    return wn != 0 or on_edge


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True)
def quadrilateral_intersection(pts1, pts2, int_pts):
    num_of_inter = 0
    for i in range(4):
        if point_in_quadrilateral_v2(pts1[2 * i], pts1[2 * i + 1], pts2):
            int_pts[num_of_inter * 2] = pts1[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1]
            num_of_inter += 1
        if point_in_quadrilateral_v2(pts2[2 * i], pts2[2 * i + 1], pts1):
            int_pts[num_of_inter * 2] = pts2[2 * i]
            int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(4):
        for j in range(4):
            has_pts = line_segment_intersection(pts1, pts2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter

@cuda.jit('(float32, float32, float32[:])', device=True, inline=True, debug=False)
def point_in_triangle(pt_x, pt_y, vertices):
    v0 = vertices[0:2]
    v1 = vertices[2:4]
    v2 = vertices[4:6]

    d0 = (pt_x - v0[0]) * (v1[1] - v0[1]) - (pt_y - v0[1]) * (v1[0] - v0[0])
    d1 = (pt_x - v1[0]) * (v2[1] - v1[1]) - (pt_y - v1[1]) * (v2[0] - v1[0])
    d2 = (pt_x - v2[0]) * (v0[1] - v2[1]) - (pt_y - v2[1]) * (v0[0] - v2[0])

    has_neg = (d0 < 0) or (d1 < 0) or (d2 < 0)
    has_pos = (d0 > 0) or (d1 > 0) or (d2 > 0)

    return not (has_neg and has_pos)


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_triangle(pts1, pts2, i, j, temp_pts):
    A = cuda.local.array((2,), dtype=numba.float32)
    B = cuda.local.array((2,), dtype=numba.float32)
    C = cuda.local.array((2,), dtype=numba.float32)
    D = cuda.local.array((2,), dtype=numba.float32)

    A[0] = pts1[2 * i]
    A[1] = pts1[2 * i + 1]

    B[0] = pts1[2 * ((i + 1) % 3)]
    B[1] = pts1[2 * ((i + 1) % 3) + 1]

    C[0] = pts2[2 * j]
    C[1] = pts2[2 * j + 1]

    D[0] = pts2[2 * ((j + 1) % 3)]
    D[1] = pts2[2 * ((j + 1) % 3) + 1]
    BA0 = B[0] - A[0]
    BA1 = B[1] - A[1]
    DA0 = D[0] - A[0]
    CA0 = C[0] - A[0]
    DA1 = D[1] - A[1]
    CA1 = C[1] - A[1]
    acd = DA1 * CA0 > CA1 * DA0
    bcd = (D[1] - B[1]) * (C[0] - B[0]) > (C[1] - B[1]) * (D[0] - B[0])
    if acd != bcd:
        abc = CA1 * BA0 > BA1 * CA0
        abd = DA1 * BA0 > BA1 * DA0
        if abc != abd:
            DC0 = D[0] - C[0]
            DC1 = D[1] - C[1]
            ABBA = A[0] * B[1] - B[0] * A[1]
            CDDC = C[0] * D[1] - D[0] * C[1]
            DH = BA1 * DC0 - BA0 * DC1
            Dx = ABBA * DC0 - BA0 * CDDC
            Dy = ABBA * DC1 - BA1 * CDDC
            temp_pts[0] = Dx / DH
            temp_pts[1] = Dy / DH
            if (temp_pts[0] == A[0] and temp_pts[1] == A[1]) or \
                    (temp_pts[0] == B[0] and temp_pts[1] == B[1]) or \
                    (temp_pts[0] == C[0] and temp_pts[1] == C[1]) or \
                    (temp_pts[0] == D[0] and temp_pts[1] == D[1]):
                return False
            return True
    return False


@cuda.jit(
    '(float32[:], float32[:], int32, int32, float32[:])',
    device=True,
    inline=True)
def line_segment_intersection_triangle_v2(pts1, pts2, i, j, temp_pts):
    a = cuda.local.array((2,), dtype=numba.float32)
    b = cuda.local.array((2,), dtype=numba.float32)
    c = cuda.local.array((2,), dtype=numba.float32)
    d = cuda.local.array((2,), dtype=numba.float32)

    a[0] = pts1[2 * i]
    a[1] = pts1[2 * i + 1]

    b[0] = pts1[2 * ((i + 1) % 3)]
    b[1] = pts1[2 * ((i + 1) % 3) + 1]

    c[0] = pts2[2 * j]
    c[1] = pts2[2 * j + 1]

    d[0] = pts2[2 * ((j + 1) % 3)]
    d[1] = pts2[2 * ((j + 1) % 3) + 1]

    area_abc = trangle_area(a, b, c)
    area_abd = trangle_area(a, b, d)

    if area_abc * area_abd >= 0:
        return False

    area_cda = trangle_area(c, d, a)
    area_cdb = area_cda + area_abc - area_abd

    if area_cda * area_cdb >= 0:
        return False
    t = area_cda / (area_abd - area_abc)

    dx = t * (b[0] - a[0])
    dy = t * (b[1] - a[1])
    temp_pts[0] = a[0] + dx
    temp_pts[1] = a[1] + dy
    if (temp_pts[0] == a[0] and temp_pts[1] == a[1]) or \
            (temp_pts[0] == b[0] and temp_pts[1] == b[1]) or \
            (temp_pts[0] == c[0] and temp_pts[1] == c[1]) or \
            (temp_pts[0] == d[0] and temp_pts[1] == d[1]):
        return False
    return True


@cuda.jit('(float32[:], float32[:], float32[:])', device=True, inline=True, debug=False)
def triangle_intersection(triangle1, triangle2, int_pts):
    num_of_inter = 0
    for i in range(3):
        if point_in_triangle(triangle1[2 * i], triangle1[2 * i + 1], triangle2):
            int_pts[num_of_inter * 2] = triangle1[2 * i]
            int_pts[num_of_inter * 2 + 1] = triangle1[2 * i + 1]
            num_of_inter += 1
        if point_in_triangle(triangle2[2 * i], triangle2[2 * i + 1], triangle1):
            int_pts[num_of_inter * 2] = triangle2[2 * i]
            int_pts[num_of_inter * 2 + 1] = triangle2[2 * i + 1]
            num_of_inter += 1
    temp_pts = cuda.local.array((2,), dtype=numba.float32)
    for i in range(3):
        for j in range(3):
            has_pts = line_segment_intersection_triangle(triangle1, triangle2, i, j, temp_pts)
            if has_pts:
                int_pts[num_of_inter * 2] = temp_pts[0]
                int_pts[num_of_inter * 2 + 1] = temp_pts[1]
                num_of_inter += 1

    return num_of_inter



@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def rbbox_to_corners(corners, rbbox):
    angle = rbbox[4]
    a_cos = math.cos(angle)
    a_sin = math.sin(angle)
    center_x = rbbox[0]
    center_y = rbbox[1]
    x_d = rbbox[2]
    y_d = rbbox[3]
    corners_x = cuda.local.array((4,), dtype=numba.float32)
    corners_y = cuda.local.array((4,), dtype=numba.float32)
    corners_x[0] = -x_d / 2
    corners_x[1] = -x_d / 2
    corners_x[2] = x_d / 2
    corners_x[3] = x_d / 2
    corners_y[0] = -y_d / 2
    corners_y[1] = y_d / 2
    corners_y[2] = y_d / 2
    corners_y[3] = -y_d / 2
    for i in range(4):
        corners[2 *
                i] = a_cos * corners_x[i] + a_sin * corners_y[i] + center_x
        corners[2 * i
                + 1] = -a_sin * corners_x[i] + a_cos * corners_y[i] + center_y


@cuda.jit('(float32[:], float32[:])', device=True, inline=True)
def inter(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((16,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    return area(intersection_corners, num_intersection)


@cuda.jit('(float32[:], float32[:])', device=True, inline=True, debug=False)
def point_to_line(point, line):
        x0, y0 = point
        x1, y1, x2, y2 = line
        return ((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1) ** 2 / ((y2-y1)**2 + (x2-x1)**2)

@cuda.jit('(float32[:], float32[:])', device=True, inline=True, debug=False)
def inter_2(rbbox1, rbbox2):
    corners1 = cuda.local.array((8,), dtype=numba.float32)
    corners2 = cuda.local.array((8,), dtype=numba.float32)
    intersection_corners = cuda.local.array((12,), dtype=numba.float32)

    rbbox_to_corners(corners1, rbbox1)
    rbbox_to_corners(corners2, rbbox2)

    num_intersection = quadrilateral_intersection(corners1, corners2,
                                                  intersection_corners)
    sort_vertex_in_convex_polygon(intersection_corners, num_intersection)

    area_ = area(intersection_corners, num_intersection)

    # sort the corners in the following order:
            #     """
            #     3 -------- 1            1 -------- 3
            #     |          |            |          |
                
            #     |          |            |          |
            #     2 -------- 0            0 -------- 2
                            
            #                     *(0, 0)
            #         ...                    ...
            #
            #     """
    distance_list = cuda.local.array((4,), dtype=numba.float32)
    sorted_indices = cuda.local.array((4,), dtype=numba.int32)
    for i in range(4):
        distance = corners1[2 * i] ** 2 + corners1[2 * i + 1] ** 2
        distance_list[i] = distance
    for i in range(4):
        sorted_indices[i] = i
    for i in range(4):
        for j in range(i + 1, distance_list.shape[0]):
            if distance_list[i] > distance_list[j]:
                distance_list[i], distance_list[j] = distance_list[j], distance_list[i]
                sorted_indices[i], sorted_indices[j] = sorted_indices[j], sorted_indices[i]

    if corners1[2 * 1] ** 2 > corners1[2 * 2] ** 2:
        sorted_indices[1], sorted_indices[2] = sorted_indices[2], sorted_indices[1]
    
    corners1_new = cuda.local.array((8,), dtype=numba.float32)
    for i in range(4):
        corners1_new[2 * i] = corners1[2 * sorted_indices[i]]
        corners1_new[2 * i + 1] = corners1[2 * sorted_indices[i] + 1]

    distance_list_2 = cuda.local.array((4,), dtype=numba.float32)
    sorted_indices_2 = cuda.local.array((4,), dtype=numba.int32)
    for i in range(4):
        distance = corners2[2 * i] ** 2 + corners2[2 * i + 1] ** 2
        distance_list_2[i] = distance
    for i in range(4):
        sorted_indices_2[i] = i
    for i in range(4):
        for j in range(i + 1, distance_list_2.shape[0]):
            if distance_list_2[i] > distance_list_2[j]:
                distance_list_2[i], distance_list_2[j] = distance_list_2[j], distance_list_2[i]
                sorted_indices_2[i], sorted_indices_2[j] = sorted_indices_2[j], sorted_indices_2[i]
    if corners2[2 * 1] ** 2 > corners2[2 * 2] ** 2:
        sorted_indices_2[1], sorted_indices_2[2] = sorted_indices_2[2], sorted_indices_2[1]
    
    corners2_new = cuda.local.array((8,), dtype=numba.float32)
    for i in range(4):
        corners2_new[2 * i] = corners2[2 * sorted_indices_2[i]]
        corners2_new[2 * i + 1] = corners2[2 * sorted_indices_2[i] + 1]

    distance_1_01 = math.sqrt((corners1_new[0] - corners2_new[0]) ** 2 + (corners1_new[1] - corners2_new[1]) ** 2)
    distance_1_02 = math.sqrt((corners1_new[0] - corners2_new[2]) ** 2 + (corners1_new[1] - corners2_new[3]) ** 2)
    distance_1_03 = math.sqrt((corners1_new[0] - corners2_new[4]) ** 2 + (corners1_new[1] - corners2_new[5]) ** 2)
    distance_1 = min(distance_1_01, distance_1_02, distance_1_03)

    distance_2_01 = math.sqrt(point_to_line(corners1_new[2:4], corners2_new[0:4]))
    distance_2_02 = math.sqrt(point_to_line(corners1_new[2:4], corners2_new[4:8]))
    distance_2 = min(distance_2_01, distance_2_02)

    temp = cuda.local.array((4,), dtype=numba.float32)
    temp[0] = corners2_new[0]
    temp[1] = corners2_new[1]
    temp[2] = corners2_new[4]
    temp[3] = corners2_new[5]

    distance_3_01 = math.sqrt(point_to_line(corners1_new[4:6], temp))
    temp[0] = corners2_new[2]
    temp[1] = corners2_new[3]
    temp[2] = corners2_new[6]
    temp[3] = corners2_new[7]
    distance_3_02 = math.sqrt(point_to_line(corners1_new[4:6], temp))
    distance_3 = min(distance_3_01, distance_3_02)

    distance = distance_1 + distance_2 + distance_3
    return area_, distance


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True)
def devRotateIoUEval(rbox1, rbox2, criterion=-1):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter = inter(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter 


@cuda.jit('(float32[:], float32[:], int32)', device=True, inline=True, debug=False)
def devRotateIoUEval_shift_zrx(rbox1, rbox2, criterion=100):
    area1 = rbox1[2] * rbox1[3]
    area2 = rbox2[2] * rbox2[3]
    area_inter,distance = inter_2(rbox1, rbox2)
    if criterion == -1:
        return area_inter / (area1 + area2 - area_inter)/ (1 + distance)
    elif criterion == 0:
        return area_inter / area1
    elif criterion == 1:
        return area_inter / area2
    else:
        return area_inter 




@cuda.jit('(int64, int64, float32[:], float32[:], float32[:], int32)', fastmath=False)
def rotate_iou_kernel_eval(N, K, dev_boxes, dev_query_boxes, dev_iou, criterion=100):
    threadsPerBlock = 8 * 8
    row_start = cuda.blockIdx.x
    col_start = cuda.blockIdx.y
    tx = cuda.threadIdx.x
    row_size = min(N - row_start * threadsPerBlock, threadsPerBlock)
    col_size = min(K - col_start * threadsPerBlock, threadsPerBlock)
    block_boxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)
    block_qboxes = cuda.shared.array(shape=(64 * 5,), dtype=numba.float32)

    dev_query_box_idx = threadsPerBlock * col_start + tx
    dev_box_idx = threadsPerBlock * row_start + tx
    if (tx < col_size):
        block_qboxes[tx * 5 + 0] = dev_query_boxes[dev_query_box_idx * 5 + 0]
        block_qboxes[tx * 5 + 1] = dev_query_boxes[dev_query_box_idx * 5 + 1]
        block_qboxes[tx * 5 + 2] = dev_query_boxes[dev_query_box_idx * 5 + 2]
        block_qboxes[tx * 5 + 3] = dev_query_boxes[dev_query_box_idx * 5 + 3]
        block_qboxes[tx * 5 + 4] = dev_query_boxes[dev_query_box_idx * 5 + 4]
    if (tx < row_size):
        block_boxes[tx * 5 + 0] = dev_boxes[dev_box_idx * 5 + 0]
        block_boxes[tx * 5 + 1] = dev_boxes[dev_box_idx * 5 + 1]
        block_boxes[tx * 5 + 2] = dev_boxes[dev_box_idx * 5 + 2]
        block_boxes[tx * 5 + 3] = dev_boxes[dev_box_idx * 5 + 3]
        block_boxes[tx * 5 + 4] = dev_boxes[dev_box_idx * 5 + 4]
    cuda.syncthreads()
    if tx < row_size:
        for i in range(col_size):
            offset = row_start * threadsPerBlock * K + col_start * threadsPerBlock + tx * K + i
            dev_iou[offset] = devRotateIoUEval_shift_zrx(block_qboxes[i * 5:i * 5 + 5],
                                                            block_boxes[tx * 5:tx * 5 + 5], criterion)


def rotate_iou_gpu_eval_1_plus_square(boxes, query_boxes, criterion=100, device_id=0):
    """rotated box iou running in gpu. 500x faster than cpu version
    (take 5ms in one example with numba.cuda code).
    convert from [this project](
        https://github.com/hongzhenwang/RRPN-revise/tree/master/pcdet/rotation).
    
    Args:
        boxes (float tensor: [N, 5]): rbboxes. format: centers, dims, 
            angles(clockwise when positive)
        query_boxes (float tensor: [K, 5]): [description]
        device_id (int, optional): Defaults to 0. [description]
    
    Returns:
        [type]: [description]
    """
    box_dtype = boxes.dtype
    boxes = boxes.astype(np.float32)
    query_boxes = query_boxes.astype(np.float32)
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    iou = np.zeros((N, K), dtype=np.float32)
    if N == 0 or K == 0:
        return iou
    threadsPerBlock = 8 * 8
    cuda.select_device(device_id)
    blockspergrid = (div_up(N, threadsPerBlock), div_up(K, threadsPerBlock))

    stream = cuda.stream()
    with stream.auto_synchronize():
        boxes_dev = cuda.to_device(boxes.reshape([-1]), stream)
        query_boxes_dev = cuda.to_device(query_boxes.reshape([-1]), stream)
        iou_dev = cuda.to_device(iou.reshape([-1]), stream)
        rotate_iou_kernel_eval[blockspergrid, threadsPerBlock, stream](
            N, K, boxes_dev, query_boxes_dev, iou_dev, criterion)
        iou_dev.copy_to_host(iou.reshape([-1]), stream=stream)
    return iou.astype(boxes.dtype)
