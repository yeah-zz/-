"""Reset selected artwork progress after Q, then open the normal RUN prompt."""
import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

from painting_session import session_lock, save
from project_paths import ROBOT_LOG_DIR
from run_json_painting import load_artwork, manifest_for, run_file


def reset_selected_progress(sources, folder=ROBOT_LOG_DIR, *, controller_stopped_confirmed=False):
    folder = Path(folder)
    # This guard remains held while P is waiting for C/Q, so reset is refused
    # until the drawing program has actually exited.
    with session_lock(folder):
        if (folder/'blue_wash_motion.lock').exists():
            raise ValueError('上色程序尚未退出，请先 P/Q 保存退出。')
        selected = []
        for source in dict.fromkeys(str(Path(s).resolve()) for s in sources):
            key = hashlib.sha256(source.casefold().encode('utf-8')).hexdigest()[:24]
            path = folder/('painting_session_'+key+'.json')
            if not path.exists():
                continue
            record = json.loads(path.read_text(encoding='utf-8'))
            if str(Path(record['source']).resolve()).casefold() != source.casefold():
                raise ValueError('绘画进度来源不匹配，未重置。')
            if (record.get('active') or record.get('status') in ('running','uncertain')) and not controller_stopped_confirmed:
                raise ValueError('该画作未正常保存退出，可能有半笔，请核查后再重置。')
            if not record.get('abandoned'):
                selected.append((path,record))
        for path,record in selected:
            record['abandoned'] = dict(
                abandoned_at=datetime.now().astimezone().isoformat(),
                reason='Operator requested a fresh sheet and restart of this artwork',
                controller_stopped_confirmed=controller_stopped_confirmed,
                previous_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
            save(path,record)
        return len(selected)


def main(argv=None):
    from launcher import setup_console
    setup_console()
    parser = argparse.ArgumentParser(description='退出绘画后重置指定画作进度，再从第一笔开始。')
    parser.add_argument('--job',required=True)
    parser.add_argument('--previous',help='同一幅画的上一个 JSON 路径，仅重置它的进度')
    args = parser.parse_args(argv)
    try:
        artwork = load_artwork(args.job)
        manifest_for(artwork[2],artwork[1],artwork[3])
        print('重新开始这幅画；会重置本幅进度并保留旧日志。')
        print('请先在旧绘画终端按 Q 退出并换新纸；此入口不会停止或移动机械臂。')
        sources=[p for p in (args.previous,args.job) if p]
        uncertain=False
        for source in sources:
            key=hashlib.sha256(str(Path(source).resolve()).casefold().encode('utf-8')).hexdigest()[:24]
            progress=ROBOT_LOG_DIR/('painting_session_'+key+'.json')
            if progress.exists():
                record=json.loads(progress.read_text(encoding='utf-8'))
                uncertain |= bool(not record.get('abandoned') and
                                  (record.get('active') or record.get('status') in ('running','uncertain')))
        if (ROBOT_LOG_DIR/'painting_session.lock').exists():
            raise ValueError('旧绘画程序仍在运行，请先在原终端 P/Q 保存退出。')
        confirmed=False
        if uncertain:
            print('旧程序未正常保存退出；请用控制器确认机械臂停止并取消旧队列，笔已离纸，换好新纸。')
            if input('上述已完成，输入 NEW 重置本幅进度；其他输入取消：').strip()!='NEW':
                return 0
            confirmed=True
        count = reset_selected_progress(sources,controller_stopped_confirmed=confirmed)
        print('已重置 %d 份本幅绘画进度；接下来输入 RUN 才从第一笔开始。' % count)
        return run_file(args.job)
    except (OSError,ValueError,KeyError,TypeError,EOFError,KeyboardInterrupt) as exc:
        print('未开始绘画：'+str(exc))
        print('如果只是按了 P，请在原终端继续按 Q 保存退出后再运行。')
        return 2


if __name__=='__main__':
    raise SystemExit(main())
