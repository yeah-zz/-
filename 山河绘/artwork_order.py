"""Explicit subject order and a final black-ink title pass."""
import copy


def ink_stages(ink, plan):
    strokes = ink['meta']['ink_execution']['base_strokes']
    if not (plan or {}).get('execution_order', {}).get('title_last'):
        return {'black': strokes}
    title = [s for s in strokes if s.get('role') == 'title']
    body = [s for s in strokes if s.get('role') != 'title']
    if not body or not title:
        raise ValueError('最后题字画稿需要山水勾勒和题字笔触。')
    return {'black': body, 'title': title}


def ink_payload(ink, strokes):
    """Regenerate safe absolute moves for a subset; preserve each exact stroke."""
    result = copy.deepcopy(ink)
    meta = result['meta']
    meta['ink_execution']['base_strokes'] = copy.deepcopy(strokes)
    meta['strokes'] = len(strokes)
    from ink_control import InkPolicy, build_plan
    from geometry import CanvasMap
    from shuimo import ink_palette_index
    from run_mountain_v4_robot import safe_transfers
    import job
    mapper = CanvasMap(meta['image_size'])
    steps, _ = build_plan(strokes, ink_palette_index(), mapper.mm_per_px,
                          InkPolicy(**meta['ink_execution']['policy']))
    points, _ = job.steps_to_waypoints(steps, mapper)
    result['waypoints'] = safe_transfers(points)
    from run_mountain_balanced import load_reviewed
    load_reviewed(result)
    return result


def available_colors(completed, counts):
    colors = [key for key in counts if key not in ('black', 'title') and completed.get(key, 0) < counts[key]]
    if colors:
        return colors
    return ['title'] if completed.get('title', 0) < counts.get('title', 0) else []


def available_stages(completed, counts):
    """Independent web stages; skipped stages never count as completed."""
    return [key for key, total in counts.items() if completed.get(key, 0) < total]
