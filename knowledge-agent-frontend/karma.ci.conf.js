module.exports = function (config) {
  config.set({
    basePath: '',
    frameworks: ['jasmine', '@angular-devkit/build-angular'],
    plugins: [require('karma-jasmine'), require('karma-chrome-launcher'), require('@angular-devkit/build-angular/plugins/karma')],
    reporters: ['progress'],
    browsers: ['CI'],
    customLaunchers: { CI: { base: 'ChromeHeadless', flags: ['--no-sandbox', '--disable-gpu'] } },
    singleRun: true, restartOnFileChange: false,
  });
};
