"""Compact, escaped Telegram result blocks with independently valid HTML chunks."""
from html import escape

from translator import LANGUAGES, LANGUAGE_FLAGS


LIMIT = 3500


def _units(text):
    return len(text.encode('utf-16-le')) // 2


def format_sections(sections):
    """Pack (heading, plain text) pairs; split only between original characters."""
    chunks, message = [], ''
    for heading, body in sections:
        if not body:
            continue
        header = '<b>' + escape(heading, quote=False) + '</b>\n'
        header_size = _units(header)
        if header_size > LIMIT - 6:
            raise ValueError('Section heading is too long.')
        parts, size = [header], header_size
        for char in body:
            encoded = escape(char, quote=False)
            cost = _units(encoded)
            if size + cost > LIMIT:
                section = ''.join(parts)
                if message:
                    chunks.append(message)
                chunks.append(section)
                message = ''
                parts, size = [header], header_size
            parts.append(encoded)
            size += cost
        section = ''.join(parts)
        if message and _units(message) + 2 + size > LIMIT:
            chunks.append(message)
            message = ''
        message += ('\n\n' if message else '') + section
    if message:
        chunks.append(message)
    return chunks


def _language(code):
    return LANGUAGE_FLAGS[code] + ' ' + LANGUAGES[code][1]


def translation_messages(text, target):
    return format_sections([('🌐 ' + _language(target), text)])


def teacher_messages(result, target, native, lang):
    correction = 'Исправление' if lang == 'ru' else 'Correction'
    if result.get('lesson_intent') == 'explain':
        explanation = 'Правило' if lang == 'ru' else 'Rule'
    else:
        explanation = 'Почему' if lang == 'ru' else 'Why'
    return format_sections([(_language(target), result['say']),
        (_language(native), result['translation']),
        ('✍️ ' + correction, result.get('correction', '')),
        ('💡 ' + explanation, result.get('explanation', ''))])
