import { describe, it, expect, vi } from 'vitest'
import { createElement } from 'react'
import { render, screen, fireEvent } from '@testing-library/react'
import { inferAttrType, parseAttrEdit } from '../lib/attrUtils'
import { LayerInspector } from '../components/LayerInspector'
import type { OnnxNode } from '../lib/onnxTypes'

// ---- Group 1: inferAttrType ----

describe('inferAttrType (v1.0)', () => {
  it('returns int for integer numbers', () => {
    expect(inferAttrType(3)).toBe('int')
    expect(inferAttrType(0)).toBe('int')
    expect(inferAttrType(-1)).toBe('int')
  })

  it('returns float for non-integer numbers', () => {
    expect(inferAttrType(3.14)).toBe('float')
    expect(inferAttrType(0.001)).toBe('float')
  })

  it('returns array for strings starting with [', () => {
    expect(inferAttrType('[1, 2, 3]')).toBe('array')
    expect(inferAttrType('[1]')).toBe('array')
  })

  it('returns string for other strings', () => {
    expect(inferAttrType('NOTSET')).toBe('string')
    expect(inferAttrType('SAME_UPPER')).toBe('string')
    expect(inferAttrType('')).toBe('string')
  })
})

// ---- Group 2: parseAttrEdit ----

describe('parseAttrEdit (v1.0)', () => {
  it('parses int input correctly', () => {
    expect(parseAttrEdit('5', 1)).toBe(5)
    expect(parseAttrEdit('-1', 0)).toBe(-1)
  })

  it('parses float input correctly', () => {
    const result = parseAttrEdit('0.001', 1e-5)
    expect(typeof result).toBe('number')
    expect(Math.abs((result as number) - 0.001)).toBeLessThan(1e-6)
  })

  it('parses array input correctly', () => {
    expect(parseAttrEdit('[2, 2]', '[1, 1]')).toBe('[2, 2]')
    expect(parseAttrEdit('3, 3', '[1, 1]')).toBe('[3, 3]')
  })

  it('returns original on invalid int', () => {
    expect(parseAttrEdit('abc', 5)).toBe(5)
  })

  it('returns original on empty string', () => {
    expect(parseAttrEdit('', 5)).toBe(5)
  })

  it('returns original on invalid array', () => {
    expect(parseAttrEdit('[a, b]', '[1, 1]')).toBe('[1, 1]')
  })

  it('passes string values through unchanged', () => {
    expect(parseAttrEdit('SAME_UPPER', 'NOTSET')).toBe('SAME_UPPER')
  })
})

// ---- Group 3: LayerInspector attribute editing ----

const baseNode: OnnxNode = {
  id: 'conv_0',
  opType: 'Conv',
  inputs: ['x', 'W'],
  outputs: ['y'],
  attributes: { kernel_shape: '[3, 3]', group: 1, dilations: '[1, 1]' },
  paramCount: 0,
  estimatedSizeMB: 0,
}

describe('LayerInspector attribute editing (v1.0)', () => {
  it('renders attribute values as clickable text', () => {
    render(createElement(LayerInspector, { node: baseNode }))
    expect(screen.getByTestId('attr-value-kernel_shape')).toBeInTheDocument()
    expect(screen.getByTestId('attr-value-group')).toBeInTheDocument()
  })

  it('clicking attribute value shows an input with the current value', () => {
    render(createElement(LayerInspector, { node: baseNode }))
    fireEvent.click(screen.getByTestId('attr-value-group'))
    const input = screen.getByTestId('attr-input-group')
    expect(input).toBeInTheDocument()
    expect((input as HTMLInputElement).value).toBe('1')
  })

  it('blurring the input calls onAttrEdit with parsed value', () => {
    const onAttrEdit = vi.fn()
    render(createElement(LayerInspector, { node: baseNode, onAttrEdit }))
    fireEvent.click(screen.getByTestId('attr-value-group'))
    const input = screen.getByTestId('attr-input-group') as HTMLInputElement
    fireEvent.change(input, { target: { value: '4' } })
    fireEvent.blur(input)
    expect(onAttrEdit).toHaveBeenCalledWith('conv_0', 'group', 4)
  })

  it('pressing Enter commits the edit', () => {
    const onAttrEdit = vi.fn()
    render(createElement(LayerInspector, { node: baseNode, onAttrEdit }))
    fireEvent.click(screen.getByTestId('attr-value-group'))
    const input = screen.getByTestId('attr-input-group') as HTMLInputElement
    fireEvent.change(input, { target: { value: '2' } })
    fireEvent.keyDown(input, { key: 'Enter' })
    expect(onAttrEdit).toHaveBeenCalledWith('conv_0', 'group', 2)
  })

  it('pressing Escape cancels without calling onAttrEdit', () => {
    const onAttrEdit = vi.fn()
    render(createElement(LayerInspector, { node: baseNode, onAttrEdit }))
    fireEvent.click(screen.getByTestId('attr-value-group'))
    const input = screen.getByTestId('attr-input-group') as HTMLInputElement
    fireEvent.change(input, { target: { value: '99' } })
    fireEvent.keyDown(input, { key: 'Escape' })
    expect(onAttrEdit).not.toHaveBeenCalled()
  })

  it('does not call onAttrEdit when value is unchanged', () => {
    const onAttrEdit = vi.fn()
    render(createElement(LayerInspector, { node: baseNode, onAttrEdit }))
    fireEvent.click(screen.getByTestId('attr-value-group'))
    const input = screen.getByTestId('attr-input-group') as HTMLInputElement
    fireEvent.change(input, { target: { value: '1' } })
    fireEvent.blur(input)
    expect(onAttrEdit).not.toHaveBeenCalled()
  })

  it('shows Modified badge when node.isModified is true', () => {
    const modNode: OnnxNode = { ...baseNode, isModified: true }
    render(createElement(LayerInspector, { node: modNode }))
    expect(screen.getByText('Modified')).toBeInTheDocument()
  })

  it('does not show Modified badge when node.isModified is false', () => {
    render(createElement(LayerInspector, { node: baseNode }))
    expect(screen.queryByText('Modified')).not.toBeInTheDocument()
  })
})
