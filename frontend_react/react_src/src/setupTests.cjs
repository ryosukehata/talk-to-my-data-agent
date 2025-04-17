// jest-dom adds custom jest matchers for asserting on DOM nodes.
require('@testing-library/jest-dom');

// Polyfill for TextEncoder/TextDecoder
const { TextEncoder, TextDecoder } = require('util');
global.TextEncoder = TextEncoder;
global.TextDecoder = TextDecoder;

// Simple mock for react-router-dom
jest.mock('react-router-dom', () => {
  return {
    useNavigate: jest.fn(() => jest.fn()),
    useParams: jest.fn(() => ({})),
    useLocation: jest.fn(() => ({ pathname: '/' })),
    Link: function Link(props) {
      return React.createElement('a', { href: props.to, ...props }, props.children);
    }
  };
});

// Mock IntersectionObserver which isn't available in test environment
class MockIntersectionObserver {
  constructor(callback) {
    this.callback = callback;
  }
  
  observe = jest.fn();
  unobserve = jest.fn();
  disconnect = jest.fn();
}

global.IntersectionObserver = MockIntersectionObserver;

// Set up to suppress console errors
const originalConsoleError = console.error;
console.error = function() {
  // Filter out specific React-related errors that are expected in test environment
  if (
    arguments[0] && 
    typeof arguments[0] === 'string' && 
    (arguments[0].includes('Warning: ReactDOM.render') || 
     arguments[0].includes('Warning: An update to'))
  ) {
    return;
  }
  
  originalConsoleError.apply(console, arguments);
};