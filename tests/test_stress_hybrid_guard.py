"""Skip only Silero work whose result cannot change RUAccent's output."""
import subprocess
import sys
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stress import BilingualAccentor


@pytest.mark.parametrize('dictionary',[{}, {'замок':['з+амок','зам+ок']}, {'привет':[]}])
def test_no_homographs_preserve_exact_ruaccent_output_without_silero_call(dictionary):
    accentor=BilingualAccentor()
    calls=[]
    accentor.silero=lambda text:calls.append(text) or 'unexpected replacement'
    text='Прив+ет, м+ир!\nЭлектро+удочник 🙂'
    assert accentor._silero_homographs(SimpleNamespace(omographs=dictionary),text)==text
    assert calls==[]


@pytest.mark.parametrize('word,result',[('з+амок','зам+ок'),('З+АМОК','ЗАМ+ОК'),('З+амок','Зам+ок')])
def test_homographs_keep_full_context_and_case_when_calling_silero(word,result):
    accentor=BilingualAccentor()
    calls=[]
    text=f'Стар+инный {word} сто+ит на холм+е.'
    expected=f'Стар+инный {result} сто+ит на холм+е.'
    accentor.silero=lambda full_text:calls.append(full_text) or expected
    ru=SimpleNamespace(omographs={'замок':['з+амок','зам+ок']})
    assert accentor._silero_homographs(ru,text)==expected
    assert calls==[text.replace('+','')]


def test_ruaccent_still_processes_no_homograph_clause():
    accentor=BilingualAccentor();calls=[]
    ru=SimpleNamespace(omographs={},process_all=lambda text:calls.append(text) or 'Прив+ет, м+ир!')
    accentor._ensure_ru=lambda:ru
    accentor.silero=lambda text:pytest.fail('No homograph can use this inference')
    assert accentor._accent_russian('Привет, мир!')=='Прив+ет, м+ир!'
    assert calls==['Привет, мир!']


def test_no_homograph_does_not_import_or_load_silero_in_cold_process():
    root=Path(__file__).resolve().parents[1]
    script="""
import sys
from types import SimpleNamespace
from stress import BilingualAccentor
attempts=[]
class WatchImport:
    def find_spec(self,fullname,path=None,target=None):
        if fullname == 'silero_stress':
            attempts.append(fullname)
            raise ImportError('Silero must stay unloaded')
sys.meta_path.insert(0,WatchImport())
assert BilingualAccentor._silero is None
assert BilingualAccentor()._silero_homographs(SimpleNamespace(omographs={}), 'plain text') == 'plain text'
assert attempts == [], attempts
"""
    result=subprocess.run([sys.executable,'-B','-c',script],cwd=root,
        env={**os.environ,'PYTHONPATH':os.pathsep.join((str(root/'voice'),str(root/'core')))},
        capture_output=True,text=True,timeout=30)
    assert result.returncode==0,result.stderr
