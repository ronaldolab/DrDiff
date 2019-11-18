# v1: A very simple Flask Hello World app to get started with...
# v2: First form with index.html
# v3: login/logout page
# v4: sum page with bootstrap layout
# v5: DrDiff is on the way, form is prepared
# v6: Deploy in Google Cloud App Engine

import sys
path = '/tmp/'
#if path not in sys.path:
#    sys.path.append(path)

# For runId (# TODO: Implement runId in mysql db)
import random

import os
from flask              import Flask, redirect, render_template, request, url_for, send_from_directory
#from flask_sqlalchemy   import SQLAlchemy
#from flask_login        import login_user, LoginManager, UserMixin, logout_user, login_required, current_user
from werkzeug.security  import check_password_hash
from werkzeug.utils     import secure_filename
from datetime           import datetime
#from flask_migrate      import Migrate
import zipfile

# This is under the hood
from DrDiff             import do_calculation
from plot_bokeh         import plot_results


app = Flask(__name__)
app.config["DEBUG"] = True

# Output folder for the output files
#OUTPUT_FOLDER = path + '/outputs/'
OUTPUT_FOLDER = path

# Upload folder for the trajectory files up to 10 MB
#UPLOAD_FOLDER = path + '/trajectories/'
UPLOAD_FOLDER = path

# (# TODO: To be Implement and tested)
ALLOWED_EXTENSIONS = set(['txt', 'dat'])

MAX_CONTENT_LENGTH = 20 * 1024 * 1024

app.config['UPLOAD_FOLDER']      = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

#SQLALCHEMY_DATABASE_URI = "mysql+mysqlconnector://{username}:{password}@{hostname}/{databasename}".format(
#    username="ronaldo",
#    password="miguel123",
#    hostname="Ronaldos-MacBook-Pro-2.local",
#    databasename="comments",
#    #databasename="ronaldojunio$dummyempty",
#)

#app.config["SQLALCHEMY_DATABASE_URI"]        = SQLALCHEMY_DATABASE_URI
#app.config["SQLALCHEMY_POOL_RECYCLE"]        = 299
#app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

#db = SQLAlchemy(app)

#migrate = Migrate(app, db)

#app.secret_key = "miguel is arranging his cars very perfectly, it is serious"
#login_manager  = LoginManager()
#login_manager.init_app(app)


#class User(UserMixin, db.Model):
#
#    __tablename__   = "users"
#
#    id              = db.Column(db.Integer, primary_key=True)
#    username        = db.Column(db.String(128))
#    password_hash   = db.Column(db.String(128))
#
#
#    def check_password(self, password):
#        return check_password_hash(self.password_hash, password)
#
#
#    def get_id(self):
#        return self.username


#@login_manager.user_loader
#def load_user(user_id):
#    return User.query.filter_by(username=user_id).first()


#class Comment(db.Model):
#
#    __tablename__   = "comments"
#
#    id              = db.Column(db.Integer, primary_key=True)
#    content         = db.Column(db.String(4096))
#    posted          = db.Column(db.DateTime, default=datetime.now)
#    commenter_id    = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=True)
#    commenter       = db.relationship('User', foreign_keys=commenter_id)

#
# Index main page will performe the calculations
#   (for now)
#
@app.route('/')
def index():

    error=""

    #if not current_user.is_authenticated:
    #    return render_template("main_page.html", errors=errors)

    #return render_template("main_page.html", errors=error)

    # testing carousel
    return render_template("main_page_carousel.html", errors=error)


# Check ALLOWED_EXTENSIONS to the uploaded trajectory file
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Compress files in a zip file
def zipoutfiles(runId):

    os.chdir(OUTPUT_FOLDER)

    zip = str(runId) + '_JobID_DrDiff.zip'

    zip_file = zipfile.ZipFile(zip, 'w')
    Files_to_Compress = [str(runId) + '_Free_energy.dat',
        str(runId) + '_DQ.dat',
        str(runId) + '_VQ.dat',
        str(runId) + '_F_stoch_Q.dat',
        str(runId) + '_input_parameters.log',
        str(runId) + '_transition_times.csv']

    for file in Files_to_Compress:
        zip_file.write(file)

    zip_file.close()

    os.chdir(path)

    return zip

# DrDiff software html page
@app.route('/software/', methods=["GET","POST"])
def software():

    errors = ""

    if request.method == "POST":

        # # Old version when runId used to be written
        # 1st, get runId and increment 1 for next run
        #file_runId = path + "runId"
        #try:
        #    f = open(file_runId, 'r')
        #except (IOError) as errno:
        #    errors += 'I/O error in getting ID to run. %s' % errno

        #runId = int(f.readline())
        #f.close()

        #runId += 1

        #f = open(file_runId, 'w')
        #f.write(str(runId))
        #f.close()
        # # end of OLD

        # temporary runId from 1 to 99, semi-open range [1, 100)
        runId = random.randrange(1, 100)

        # Parse values from the parameters form

        # request Q(t) file, it is currently treated in DrDiff.do_calculation
        # Here, it is only treated the security of html side
        if 'inputFile01' not in request.files:
            errors += '<div class="alert alert-warning mt-3" role="alert">No file part!</div>'
            return render_template("software.html", errors=errors)

        file = request.files['inputFile01']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            errors += '<div class="alert alert-warning mt-3" role="alert">No selected file!</div>'
            return render_template("software.html", errors=errors)

        # File and extension are ok to pass
        if file: #and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            # Treat parameters from input form

            #beta        = float(request.form["beta"])
            #Eq          = int(request.form["EquilibrationSteps"])
            #Q_zero      = float(request.form["Q_zero"])
            #Q_one       = float(request.form["Q_one"])
            #Qbins       = float(request.form["Qbins"])
            #timestep   = float(request.form["timestep"])
            #Snapshot    = int(request.form["Snapshot"])
            #tmin        = int(request.form["tmin"])
            #tmax        = int(request.form["tmax"])

            try:
                beta = float(request.form["beta"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for beta.</div>'.format(request.form["beta"])
                return render_template("software.html", errors=errors)

            try:
                EquilibrationSteps = int(request.form["EquilibrationSteps"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for EquilibrationSteps.</div>'.format(request.form["EquilibrationSteps"])
                return render_template("software.html", errors=errors)

            try:
                Q_zero = float(request.form["Q_zero"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for Q_zero.</div>'.format(request.form["Q_zero"])
                return render_template("software.html", errors=errors)

            try:
                Q_one = float(request.form["Q_one"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for Q_one.</div>'.format(request.form["Q_one"])
                return render_template("software.html", errors=errors)

            try:
                Qbins = float(request.form["Qbins"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for Qbins.</div>'.format(request.form["Qbins"])
                return render_template("software.html", errors=errors)

            try:
                timestep = float(request.form["timestep"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for timestep.</div>'.format(request.form["timestep"])
                return render_template("software.html", errors=errors)

            try:
                Snapshot = int(request.form["Snapshot"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for Snapshot.</div>'.format(request.form["Snapshot"])
                return render_template("software.html", errors=errors)

            try:
                tmin = int(request.form["tmin"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for tmin.</div>'.format(request.form["tmin"])
                return render_template("software.html", errors=errors)

            try:
                tmax = int(request.form["tmax"])
            except:
                errors += '<div class="alert alert-warning mt-3" role="alert">{!r} is not a valid number for tmax.</div>'.format(request.form["tmax"])
                return render_template("software.html", errors=errors)

            # Must be Q_zero < Q_one
            if Q_zero > Q_one : Q_zero, Q_one = Q_one, Q_zero

            # end of treatment form

            # Now, calculate and show results
            # DrDiff.do_calculation() does all the calculation
            #result, out_traj, errorDrDiff = do_calculation(runId, path,
            #    (UPLOAD_FOLDER + filename), (OUTPUT_FOLDER  + str(runId) + "_"),
            #    beta,EquilibrationSteps,Q_zero,Q_one,Qbins,timestep,Snapshot,tmin,tmax)
            result, out_traj, errorDrDiff = do_calculation(runId, path,
                (UPLOAD_FOLDER + filename), (OUTPUT_FOLDER  + str(runId) + "_"),
                beta,EquilibrationSteps,Q_zero,Q_one,Qbins,timestep,Snapshot,tmin,tmax)

            # Compress output (data) files in a zip file
            zipfile = zipoutfiles(runId)

            # Prepare plots with bokeh models from plot_bokeh.py
            script_plot, div_plot = plot_results(runId, path, Q_zero, Q_one)

            return render_template("software.html",
                script=script_plot,
                out_traj=out_traj,
                result=div_plot,
                transition_times_dict=result,
                runId=runId,
                filename=filename,
                zipfile=zipfile,
                errors=errors)

        # File and extension not ok
        else:
            errors += '<div class="alert alert-warning mt-3" role="alert">File not with allowed extension ({!r})!</div>'.format(ALLOWED_EXTENSIONS)
            return render_template("software.html", errors=errors)

        # end of treating inputfile

    #if not current_user.is_authenticated:
    #    return render_template("software.html", errors=errors)

    #if request.method == "GET":
    return render_template("software.html", errors=errors)


# Create the route for the zipfile to be downloaded
@app.route("/outputs/<path:zipfile>", methods=['GET', 'POST'])
def getFile(zipfile):
    return send_from_directory(OUTPUT_FOLDER, zipfile, as_attachment=True)

# Create the route for the traj_Q image
@app.route("/outputs/<path:out_traj>", methods=['GET', 'POST'])
def getFileQ(out_traj):
    return send_from_directory(OUTPUT_FOLDER, out_traj)


# Definition of the comments method
#   Users can leave comments for the developers
@app.route('/comments/', methods=["GET","POST"])
def comments():
    if request.method == "GET":
        return render_template("comments.html", comments=Comment.query.all())

    if not current_user.is_authenticated:
        return redirect(url_for('comments'))

    comment = Comment(content=request.form["contents"], commenter=current_user)
    db.session.add(comment)
    db.session.commit()

    return redirect(url_for('comments'))


# @app.route("/login/", methods=["GET", "POST"])
# def login():
#     if request.method == "GET":
#         return render_template("login_page.html", error=False)
#
#     user = load_user(request.form["username"])
#     if user is None:
#         return render_template("login_page.html", error=True)
#
#     if not user.check_password(request.form["password"]):
#         return render_template("login_page.html", error=True)
#
#     login_user(user)
#
#     return redirect(url_for('index'))
#
#
# @app.route("/logout/")
# @login_required
# def logout():
#     logout_user()
#     return redirect(url_for('index'))

@app.route("/about/")
def about():
    return render_template("about.html")

@app.route("/download/")
def download():
    return render_template("download.html")

@app.route("/cite/")
def cite():
    return render_template("cite.html")

@app.route("/team/")
def team():
    return render_template("team.html")


if __name__ == '__main__':
    # This is used when running locally only. When deploying to Google App
    # Engine, a webserver process such as Gunicorn will serve the app. This
    # can be configured by adding an `entrypoint` to app.yaml.
    # Flask's development server will automatically serve static files in
    # the "static" directory. See:
    # http://flask.pocoo.org/docs/1.0/quickstart/#static-files. Once deployed,
    # App Engine itself will serve those files as configured in app.yaml.
    app.run(host='127.0.0.1', port=8080, debug=True)
