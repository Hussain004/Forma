import '@testing-library/jest-dom'

// React Flow requires ResizeObserver, which jsdom does not implement.
global.ResizeObserver = class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}
