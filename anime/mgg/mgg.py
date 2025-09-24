from manim import *
import numpy as np

class GroupGemmWithPerLayerMasks(Scene):
    def construct(self):
        # 1) 原始维度
        M, K, N = 120, 710, 400
        G_VIEW = 32
        M_VIEW = 1024
        K_VIEW = 7168
        N_VIEW = 4096  # 仅用于标签显示
        G = 4

        # 2) 原始（未缩放）斜向偏移，单位=像素意义的“原始度量”
        diag_up_raw = 1.0
        diag_right_raw = 1.2

        # 3) 视觉参数（缩放前的基础值）
        stroke_w_raw = 3.0
        label_scale = 0.7
        symbol_scale = 1.2

        # 4) 先以原始尺寸计算占位框（未缩放）
        def stacked_bbox_size_raw(width, height):
            w = width + (G - 1) * diag_right_raw
            h = height + (G - 1) * diag_up_raw
            return w, h

        Aw_raw, Ah_raw = stacked_bbox_size_raw(K, M)   # A: width=K, height=M
        Bw_raw, Bh_raw = stacked_bbox_size_raw(N, K)   # B: width=N, height=K
        Cw_raw, Ch_raw = stacked_bbox_size_raw(N, M)   # C: width=N, height=M

        # 5) 布局是 L 形：A 左，B 在右上，C 在右下
        #    我们估计整体包围盒，给出间距预算；随后根据屏幕计算统一缩放 s
        inter_block_gap_raw = 60.0
        vertical_gap_raw = 40.0  # B 提升 & C 下移的冗余

        total_width_raw = Aw_raw + inter_block_gap_raw + max(Bw_raw, Cw_raw)
        total_height_raw = max(Ah_raw + vertical_gap_raw + Ch_raw, Bh_raw + vertical_gap_raw + Ah_raw)

        # 6) 根据屏幕可用大小计算缩放系数 s
        #    典型 16:9 屏幕宽≈14.22，高≈8；留一定边距
        screen_w = config.frame_width
        screen_h = config.frame_height
        pad = 0.8  # 使用 80% 空间，留边距给标签与符号
        target_w = screen_w * pad
        target_h = screen_h * pad

        s_w = target_w / total_width_raw
        s_h = target_h / total_height_raw
        s = min(s_w, s_h)

        # 7) 将所有度量按 s 缩放
        diag_up = diag_up_raw * s
        diag_right = diag_right_raw * s
        inter_block_gap = inter_block_gap_raw * s * 2
        vertical_gap = vertical_gap_raw * s
        stroke_w = max(1.5, stroke_w_raw * s)  # 线宽不要太细
        label_buff = 0.35
        symbol_h_gap = -0.20
        symbol_v_offset = 0.35

        # 让整体略微居中：A 放左下方一点点即可，其余 next_to/shift 计算
        origin_offset = LEFT * (total_width_raw * s / 2 - Aw_raw * s / 2) * 1.8 + DOWN * 1.7

        # 8) 创建基础矩形（已缩放尺寸）
        A_base = Rectangle(height=M * s, width=K * s, color=BLUE, fill_opacity=1.0).set_stroke(width=stroke_w)
        B_base = Rectangle(height=K * s, width=N * s, color=GREEN, fill_opacity=1.0).set_stroke(width=stroke_w)
        C_base = Rectangle(height=M * s, width=N * s, color=RED, fill_opacity=1.0).set_stroke(width=stroke_w)

        # 9) 占位框（已缩放）
        def stacked_bbox_size(rect):
            w = rect.width + (G - 1) * diag_right
            h = rect.height + (G - 1) * diag_up
            return w, h

        Aw, Ah = stacked_bbox_size(A_base)
        Bw, Bh = stacked_bbox_size(B_base)
        Cw, Ch = stacked_bbox_size(C_base)

        def make_placeholder(w, h):
            return Rectangle(width=w, height=h).set_opacity(0).set_stroke(width=0)

        A_ph = make_placeholder(Aw, Ah)
        B_ph = make_placeholder(Bw, Bh)
        C_ph = make_placeholder(Cw, Ch)

        # 10) L 形布局
        A_ph.move_to(ORIGIN + origin_offset)
        B_ph.next_to(A_ph, RIGHT, buff=inter_block_gap * 3)
        # 让 B 顶对齐到 A 顶附近，留一些垂直间隙
        B_raise = (A_ph.get_top()[1] - B_ph.get_bottom()[1]) + vertical_gap * 0.5
        B_ph.shift(UP * B_raise)

        C_ph.next_to(A_ph, RIGHT, buff=inter_block_gap * 3)
        # 让 C 底在 A 下方，留一些间隙，且与 B 不重叠
        C_lower = (C_ph.get_top()[1] - B_ph.get_bottom()[1]) + vertical_gap * 0.5
        C_ph.shift(DOWN * C_lower)

        # 11) 将基础矩形对齐到占位框左下角
        def align_base_to_placeholder(base_rect, ph_rect):
            ph_lb = ph_rect.get_corner(DOWN + LEFT)
            base_lb = base_rect.get_corner(DOWN + LEFT)
            base_rect.shift(ph_lb - base_lb)
            return base_rect

        align_base_to_placeholder(A_base, A_ph)
        align_base_to_placeholder(B_base, B_ph)
        align_base_to_placeholder(C_base, C_ph)

        # 12) 标签与符号
        label_A = Text(f"A [{M_VIEW}×{K_VIEW}]").scale(label_scale).next_to(A_ph, UP, buff=label_buff)
        label_B = Text(f"B [{K_VIEW}×{N_VIEW}]").scale(label_scale).next_to(B_ph, LEFT, buff=label_buff)
        label_C = Text(f"C [{M_VIEW}×{N_VIEW}]").scale(label_scale).next_to(C_ph, DOWN, buff=label_buff + 0.1)

        multiply_sign = Text("×").scale(symbol_scale)
        mid_AB = (A_ph.get_right() + B_ph.get_left()) / 2
        multiply_sign.move_to(mid_AB + UP * symbol_v_offset + RIGHT * symbol_h_gap * 0.2)

        equals_sign = Text("=").scale(symbol_scale)
        mid_AC = (A_ph.get_right() + C_ph.get_left()) / 2
        equals_sign.move_to(mid_AC + RIGHT * symbol_h_gap * 0.2)

        # 13) 初始绘制普通 GEMM
        self.play(Create(A_base), Write(label_A))
        self.play(Create(B_base), Write(label_B), Write(multiply_sign))
        self.play(Create(C_base), Write(label_C), Write(equals_sign))
        self.wait(0.25)

        # 14) 构建斜向堆叠层
        def build_diagonal_layers(base_rect, G, fill_color, edge_color):
            layers = VGroup()
            for i in range(G):
                r = base_rect.copy()
                r.set_fill(fill_color, opacity=1.0)
                r.set_stroke(color=edge_color, width=stroke_w, opacity=1.0)
                r.shift(UP * (diag_up * i * 10) + RIGHT * (diag_right * i * 10))
                layers.add(r)
            return layers

        A_layers = build_diagonal_layers(A_base, G, BLUE, BLUE_D)
        B_layers = build_diagonal_layers(B_base, G, GREEN, GREEN_D)
        C_layers = build_diagonal_layers(C_base, G, RED, RED_D)

        # 15) 普通 → Group GEMM
        self.play(
            ReplacementTransform(A_base, A_layers, path_arc=0.15),
            ReplacementTransform(B_base, B_layers, path_arc=0.15),
            ReplacementTransform(C_base, C_layers, path_arc=0.15),
            run_time=1.0
        )

        # 16) 标签更新
        self.play(
            Transform(label_A, Text(f"A [{G_VIEW}×{M_VIEW}×{K_VIEW}]").scale(label_scale).move_to(label_A)),
            Transform(label_B, Text(f"B [{G_VIEW}×{K_VIEW}×{N_VIEW}]").scale(label_scale).move_to(label_B)),
            Transform(label_C, Text(f"C [{G_VIEW}×{M_VIEW}×{N_VIEW}]").scale(label_scale).move_to(label_C)),
            run_time=0.8
        )

        # 17) 掩码逻辑
        masked_rows_per_group = {
            0: list(range(30)),
            1: list(range(60)),
            2: list(range(20)),
            3: list(range(80)),
        }

        def make_layer_container_with_masks(layer_rect, rows, M_rows, mask_color=GREY_B, stroke_color=YELLOW, hl_opacity=0.45, edge_opacity=0.9, dim=False):
            container = VGroup()
            container.add(layer_rect)

            row_h = layer_rect.height / M_rows
            bottom_y = layer_rect.get_bottom()[1]
            cx = layer_rect.get_center()[0]

            for r in rows:
                if 0 <= r < M_rows:
                    y_center = bottom_y + r * row_h + row_h / 2
                    mask_bar = Rectangle(
                        width=layer_rect.width,
                        height=row_h,
                        fill_color=mask_color,
                        fill_opacity=hl_opacity,
                        stroke_color=stroke_color,
                        stroke_width=max(1.0, 2.0 * s),
                        stroke_opacity=0.8
                    ).move_to([cx, y_center, 0])
                    container.add(mask_bar)

                    if dim:
                        dim_cover = Rectangle(
                            width=layer_rect.width,
                            height=row_h,
                            fill_color=BLACK,
                            fill_opacity=0.18,
                            stroke_opacity=0.0
                        ).move_to([cx, y_center, 0])
                        container.add(dim_cover)

            return container

        A_layer_containers = VGroup()
        C_layer_containers = VGroup()
        for g in range(G):
            rows = masked_rows_per_group.get(g, [])
            A_layer_containers.add(
                make_layer_container_with_masks(A_layers[g], rows, M_rows=M, mask_color=GREY_B, stroke_color=BLUE_D, hl_opacity=0.1, dim=False)
            )
            C_layer_containers.add(
                make_layer_container_with_masks(C_layers[g], rows, M_rows=M, mask_color=GREY_B, stroke_color=RED_D, hl_opacity=0.1, dim=True)
            )

        self.remove(A_layers, C_layers)
        self.add(A_layer_containers, C_layer_containers)

        for i in range(G):
            self.add(A_layer_containers[i])
            self.add(C_layer_containers[i])

        anims_A = []
        anims_C = []
        for layerA, layerC in zip(A_layer_containers, C_layer_containers):
            mA = VGroup(*layerA[1:])
            mC = VGroup(*layerC[1:])
            if len(mA) > 0:
                anims_A.append(FadeIn(mA, lag_ratio=0.1, run_time=0.4))
            if len(mC) > 0:
                anims_C.append(FadeIn(mC, lag_ratio=0.1, run_time=0.4))
        if anims_A or anims_C:
            self.play(*anims_A, *anims_C)

        self.wait(2)
