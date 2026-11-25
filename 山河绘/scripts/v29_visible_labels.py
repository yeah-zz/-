# -*- coding: utf-8 -*-
"""Assistant-visible observations on original patches, before v29 training.

Labels describe permissible trajectory continuation, not historical pen lifts.
Unlisted pairs stay unknown. Invalid branches are graph spurs/loops with no
separately visible ink line; relations involving them are non-continuations.
"""
ANNOT={
 'T000':dict(note='淡线与分支编号重叠，无法可靠判定，全部未知'),
 'T001':dict(invalid=[1,2],note='尖笔末端小环没有独立可见墨线，不应把主线续入小环'),
 'T002':dict(same=[(0,2)],no_connect=[(0,1),(1,2)],note='左侧折曲长线0接2，右上细线1作为独立短线'),
 'T003':dict(note='三条细线交汇走向不充分明确，全部未知'),
 'T004':dict(invalid=[1,2],note='粗笔末端内部小环缺乏可见独立墨线'),
 'T005':dict(duplicate_of='T004',note='与T004为同一画作的重复局部，排除重复计数'),
 'T006':dict(invalid=[1,2],note='竖向干笔末端环状分支不对应独立画线'),
 'T007':dict(note='斜线与轮廓并接处无法可靠确定同笔，全部未知'),
 'T008':dict(same=[(0,3)],no_connect=[(0,1),(0,2),(1,3),(2,3)],note='右侧长边0接3，左侧短线与长边分开；1和2关系未知'),
 'T009':dict(invalid=[0,1],note='宽弯笔内部小环不作为单独笔画续接'),
 'T010':dict(note='宽竖笔多侧缘混在一起，中心与分笔不确定'),
 'T011':dict(note='岩石密集交汇与墨团，全部未知'),
 'T012':dict(duplicate_of='T010',note='与T010为同一画作重复局部，排除'),
 'T013':dict(invalid=[2,3],note='宽弯笔内部短环缺少独立墨线；0和1真实走向仍未知'),
 'T014':dict(same=[(0,1)],invalid=[2],note='山石外沿0接1，向空白偏出的短支2不续接'),
 'T015':dict(note='密集短皴线交叠，无法可靠判别同笔'),
 'T016':dict(note='复杂浅岩石与粗墨，全部未知'),
 'T017':dict(same=[(0,1)],no_connect=[(0,2),(1,2)],note='小石外轮廓0接1，内部短线2分开'),
 'T018':dict(note='局部可能是树干而非山石，笔触重叠无法可靠判别'),
 'T019':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],note='左侧外沿1接底部弯线2，斜入的粗笔0分开'),
 'T020':dict(same=[(0,1)],invalid=[2,3],note='底部横笔0接1，下方小环不作为独立笔画'),
 'T021':dict(duplicate_of='T020',note='与T020重复局部，排除'),
 'T022':dict(invalid=[1,2],note='粗墨内的小环没有独立画线；上方淡线是否真实仍未知'),
 'T023':dict(duplicate_of='T022',note='与T022为重复画作局部，排除'),
 'T024':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],note='前方石帽弯线1接2，上方淡线0作为另起线处理'),
 'T025':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],note='下方淡横线1接2，上方断笔0不跨空白续入横线'),
 'T026':dict(note='整块宽墨没有清楚的内部中心走向，全部未知'),
 'T027':dict(same=[(0,2)],no_connect=[(0,1),(1,2)],note='长竖笔0接2，邻近向右分出的细皴线1分开'),
 'T028':dict(note='横笔和淡竖笔的交接中心不清楚，保留未知'),
 'T029':dict(invalid=[1,2],note='宽墨内部小环缺乏独立可见墨线'),
 'V030':dict(same=[(0,2)],no_connect=[(0,1),(1,2)],note='外轮廓0接2，内侧短线1分开'),
 'V031':dict(invalid=[1,2],note='横笔进入宽墨处的小环不作独立画线'),
 'V032':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],note='弯折长线1接2，横向短线0分开'),
 'V033':dict(note='宽钩形笔的中心走向无法精确确定，全部未知'),
 'V034':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],note='底部轮廓1接2，内部皴线0分开'),
 'V035':dict(same=[(1,2)],no_connect=[(0,1),(0,2)],invalid=[3],note='石块下沿1接2，邻石边0分开，空白内短支3不续接'),
 'V036':dict(same=[(0,1)],invalid=[2],note='长弯笔1接0，向下的短侧支2没有独立笔画支撑'),
 'V037':dict(same=[(0,2)],no_connect=[(0,1),(1,2)],note='较长竖线0接2，邻接短支1分开'),
 'V038':dict(note='多条荷叶皴在宽墨处重叠，保留未知'),
 'V039':dict(same=[(0,1)],invalid=[2],note='斜笔0接1，向左偏出的支2不对应清楚墨线'),
}
